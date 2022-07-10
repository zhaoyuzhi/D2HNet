import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from easydict import EasyDict as edict

import util.parallel as P
from .utils import create_generator_val
from .network_module import GradLayer
from .vgg.vgg_module import VGG
from .pytorch_ssim import SSIM


class LossBase(P.Parallel):

    def __init__(self, num_gpus):
        P.Parallel.__init__(self, num_gpus = num_gpus)


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.L1loss = nn.L1Loss()

    def RGB2YUV(self, RGB):
        YUV = RGB.clone()
        YUV[:, 0, :, :] = 0.299 * RGB[:, 0, :, :] + 0.587 * RGB[:, 1, :, :] + 0.114 * RGB[:, 2, :, :]
        YUV[:, 1, :, :] = -0.14713 * RGB[:, 0, :, :] - 0.28886 * RGB[:, 1, :, :] + 0.436 * RGB[:, 2, :, :]
        YUV[:, 2, :, :] = 0.615 * RGB[:, 0, :, :] - 0.51499 * RGB[:, 1, :, :] - 0.10001 * RGB[:, 2, :, :]
        return YUV

    def forward(self, x, y):
        yuv_x = self.RGB2YUV(x)
        yuv_y = self.RGB2YUV(y)
        return self.L1loss(yuv_x, yuv_y)


class VGGLossBase(nn.Module):
    
    def __init__(self, vgg_model_path):
        super(VGGLossBase, self).__init__()
        self.vgg = VGG(vgg16_model_path = vgg_model_path)


class VGGLoss(VGGLossBase):

    def __init__(self, vgg_model_path, mode = 'mse'):
        super(VGGLoss, self).__init__(vgg_model_path)

        if mode == 'mse':
            self.loss = nn.MSELoss()
        elif mode == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise ValueError('mode %s not supported in VGGLoss.' % (mode))

    def forward(self, x, y):
        x_feat = self.vgg(x)[-1]
        y_feat = self.vgg(y)[-1]
        return self.loss(x_feat, y_feat)


class DeBlurVGGLoss(VGGLoss):

    def __init__(self, vgg_model_path, mode = 'mse'):
        super(DeBlurVGGLoss, self).__init__(vgg_model_path = vgg_model_path, mode = mode)


class WGANLoss(nn.Module):

    def __init__(self):
        super(WGANLoss, self).__init__()

    def forward(self, fake_out, gt_out, flag):
        """
        :param
        :param flag: G or D
        :return:
        """
        if flag == 'G':
            return -torch.mean(fake_out)
        elif flag == 'D':
            return -torch.mean(gt_out) + torch.mean(fake_out)
        else:
            raise ValueError('flag should be G or D, not %s' % flag)


class DeblurLoss(nn.Module):

    def __init__(self, mode = 'mse'):
        super(DeblurLoss, self).__init__()
        if mode == 'mse':
            self.loss = nn.MSELoss()
        elif mode == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise ValueError('mode %s not supported in DeblurLoss.' % (mode))

    def forward(self, deblur_out, gt):
        return self.loss(deblur_out, gt)


class DenoiseLoss(DeblurLoss):

    def __init__(self, mode = 'mse'):
        super(DenoiseLoss, self).__init__(mode = mode)


class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)


class DistillLossBase(nn.Module):

    def __init__(self, big_model_config, big_model_path):
        """
        big_model_config: training config of big model
        big_model_path: model path of big model
        """
        super(DistillLossBase, self).__init__()
        with open(big_model_config, 'r') as f:
            config = edict(yaml.load(f))
        self.bigModel = create_generator_val(config.GNet, big_model_path)


class DistillLoss(DistillLossBase):

    def __init__(self, big_model_config, big_model_path, big_model_output, small_model_output, 
                 output_weight, mode = 'mse'):
        """
        big_model_output: list, indicates the output index of big model, paired with small model. e.g. [0,1,1]
        small_model_output: list, indicates the output index of small model, paired with big model. e.g. [0,1,2]
        output_weight: list, weight for each output pair.
        mode: loss mode, mse or l1
        """
        super(DistillLoss, self).__init__(big_model_config, big_model_path)
        if mode.lower() == 'mse':
            self.loss = nn.MSELoss()
        elif self.mode.lower() == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise ValueError('Unsupported loss mode %s.' % mode)
        
        self.big_model_output = big_model_output
        self.small_model_output = small_model_output
        self.output_weight = output_weight
        assert len(big_model_output) == len(small_model_output) and len(big_model_output) == len(output_weight)

    def forward(self, small_outs, short_img, long_img):
        loss_array = []
        big_outs = []

        if len(self.big_model_output) > 0:
            big_model_out = self.bigModel(short_img, long_img)
            # small_model_out = self.small_model_output(short_img, long_img)
            for i in range(len(self.big_model_output)):
                big_out = big_model_out[self.big_model_output[i]]
                big_outs.append(big_out)
                loss = self.output_weight[i] * self.loss(big_out, small_outs[self.small_model_output[i]])
                loss_array.append(loss)
        
        return loss_array, big_outs
            

class AttMaskLoss(nn.Module):

    def __init__(self):
        super(AttMaskLoss, self).__init__()

    def forward(self, mask):
        return torch.mean(mask)


class CXLoss(VGGLossBase):

    def __init__(self, vgg_model_path, gt_layers, gt_weights, ref_layers, ref_weights, sigma = 0.1, b = 1.0, mode = "consine"):
        '''
        :param gt_layers/ref_layers: list, vgg layers to calculate cx loss, for gt or ref
        :param gt_weights/ref_weights: list, weights for each vgg layer, for gt or ref
        :return:
        '''
        super(CXLoss, self).__init__(vgg_model_path)
        self.gt_layers = gt_layers
        self.gt_weights = gt_weights
        self.ref_layers = ref_layers
        self.ref_weights = ref_weights
        self.mode = mode
        self.sigma = sigma
        self.b = b
        assert len(self.gt_layers) == len(self.gt_weights)
        assert len(self.ref_layers) == len(self.ref_weights)

    def rand_sampling(self, features, s = 64, d_indices = None):
        N, C, H, W = features.size()
        features = features.view(N, C, -1)
        all_indices = torch.randperm(H * W)
        select_indices = torch.arange(0, s ** 2, dtype = torch.long)
        d_indices = torch.gather(all_indices, dim = 0, index = select_indices) if d_indices is None else d_indices
        features = features[:, :, d_indices]
        re = features.contiguous().view(N, C, s, s)
        return re, d_indices

    def rand_pooling(self, features, s = 64):
        s_features = []
        sample_feature_0, d_indices = self.rand_sampling(features[0], s)
        s_features.append(sample_feature_0)
        for i in range(1, len(features)):
            sample_feature, _ = self.rand_sampling(features[i], s = s, d_indices = d_indices)
            s_features.append(sample_feature)
        return s_features 

    def cx_loss(self, x, y):
        N, C, H, W = x.size()
        mu_y = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
        x_centered = x - mu_y
        y_centered = y - mu_y
        x_normalized = x_centered / torch.norm(x_centered, p = 2, dim = 1, keepdim = True)
        y_normalized = y_centered / torch.norm(y_centered, p = 2, dim = 1, keepdim = True)

        x_normalized = x_normalized.view(N, C, -1)
        y_normalized = y_normalized.view(N, C, -1)
        consine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)

        d = 1 - consine_sim
        d_min, _ = torch.min(d, dim = 2, keepdim = True)

        d_tilde = d / (d_min + 1e-5)
        w = torch.exp((1 - d_tilde) / self.sigma)
        cx_ij = w / torch.sum(w, dim = 2, keepdim = True)
        cx = torch.mean(torch.max(cx_ij, dim = 1)[0], dim = 1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))
        return cx_loss

    def forward(self, out, gt, ref):
        '''
        :param out: network output
        :param gt:
        :param ref:
        :return:
        '''
        
        featureI = self.vgg(out)

        if gt is not None:
            featureT = self.vgg(gt)

            loss = 0
            for i in range(len(self.gt_layers)):
                featI = featureI[self.gt_layers[i]]
                featT = featureT[self.gt_layers[i]].detach()

                if featI.size(2) > 64:
                    featI, featT = self.rand_pooling([featI, featT], s = 64)

                cx_loss = self.cx_loss(featI, featT) * self.gt_weights[i]
                loss += cx_loss
        
        if ref is not None:
            featureR = self.vgg(ref)

            loss = 0
            for i in range(len(self.ref_layers)):
                featI = featureI[self.ref_layers[i]]
                featR = featureR[self.ref_layers[i]].detach()

                if featI.size(2) > 64:
                    featI, featR = self.rand_pooling([featI, featR], s = 64)

                cx_loss = self.cx_loss(featI, featR) * self.ref_weights[i]
                loss += cx_loss

        return loss


class SSIMLoss(nn.Module):

    def __init__(self, window_size = 11):
        super(SSIMLoss, self).__init__()
        self.loss = SSIM(window_size = window_size)

    def forward(self, x, y):
        """
        x: output of network
        y: gt
        """
        return self.loss(x, y)
