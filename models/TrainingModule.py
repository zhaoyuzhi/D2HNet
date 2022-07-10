import os
import numpy as np
import json
from datetime import datetime
from easydict import EasyDict as edict
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from . import loss as L
import util.parallel as P
import util.visualboard as V


def pack_network_output(out, net_name):
    if isinstance(out, torch.Tensor):
        return {'output': out}
    elif isinstance(out, list) or isinstance(out, tuple):
        output = {'output': out[0]}
        if net_name == 'AttNet':
            output['att_mask'] = out[1]
        elif net_name == 'DDNet':
            output['deblur_out'] = out[1]
            output['grad_output'] = out[2]
            output['lr_grad'] = out[3]
        elif net_name == 'DDNetV1' or net_name == 'DDNetV2' or net_name == 'DDNetV2_DWT1' or net_name == 'DDNetV2_DWT2':
            output['denoise_out'] = out[1]
        # elif net_name == 'DeblurNet':
        #     output['deblur_out'] = out[0]
        return output
    else:
        raise ValueError("network output format is illegal.")


def pack_gt_data(data: list):
    gt_dict = {}
    gt_dict['RGBout_img'] = data[0]
    gt_dict['gt_long_img'] = data[1]
    return gt_dict


class LossManager(L.LossBase):

    def __init__(self, loss_conf, num_gpus):
        super(LossManager, self).__init__(num_gpus = num_gpus)
        self.loss_conf = loss_conf
        self.criterions = {}
        self.train_GAN = False
        self.distill_training = False
        for k, v in loss_conf.items():
            print('LossManager:', k, v)
            # pytorch built-in loss
            if hasattr(nn, k):
                if hasattr(v, 'args'):
                    func = getattr(nn, k)(**(v.args))
                else:
                    func = getattr(nn, k)()
                self.criterions[k] = func.to(self.device)
            # self defined loss
            else:
                if hasattr(v, 'args'):
                    func = getattr(L, k)(**(v.args))
                else:
                    func = getattr(L, k)()
                self.criterions[k] = func.to(self.device)

            # warp pretrained model
            if isinstance(self.criterions[k], L.VGGLossBase):
                self.vgg = self.criterions[k].vgg
                self.criterions[k].vgg = self.wrapper(self.criterions[k].vgg)

            if isinstance(self.criterions[k], L.DistillLossBase):
                self.criterions[k].bigModel = self.wrapper(self.criterions[k].bigModel)

            if 'GAN' in k:
                self.train_GAN = True
            if isinstance(self.criterions[k], L.DistillLossBase):
                self.distill_training = True
    
    def __call__(self, output, gt_dict, fake_out):
        """
        :param output: output of network, packed as a dict using `pack_network_output`
        :param gt_dict: ground truth dict
        :param fake_out: DNet input
        :return:
        """

        if hasattr(self, 'vgg'):
            self.vgg.cache.clear()
            
        out = output['output']

        gt_img = gt_dict['RGBout_img']

        loss_info = {}
        for k, v in self.criterions.items():
            
            # DistillLoss will be computed in distill_loss
            if isinstance(v, L.DistillLossBase):
                continue

            if 'GAN' in k:
                loss_info[k] = v(fake_out, None, flag = 'G')
            elif k == 'DeblurLoss':
                loss_info[k] = v(output['deblur_out'], gt_img)
            elif k == 'DenoiseLoss':
                loss_info[k] = v(output['denoise_out'], gt_img)
            elif k == 'DeBlurVGGLoss':
                loss_info[k] = v(output['deblur_out'], gt_img)
            elif k == 'AttMaskLoss':
                loss_info[k] = v(output['att_mask'])
            elif k == 'CXLoss':
                loss_info[k] = v(out, gt_img, gt_dict['gt_long_img'])
            else:
                loss_info[k] = v(out, gt_img)
            '''
            elif k == 'GradLoss':
                loss_info[k] = v(output['grad_output'], gt_img)
            '''
            
        loss_sum = 0.
        for k, v in loss_info.items():
            loss_sum += self.loss_conf[k].weight * v
            loss_info[k] = round(self.loss_conf[k].weight * v.item(), 5)
        
        return loss_sum, loss_info

    def distill_loss(self, small_outs, short_img, long_img):
        """
        :param short_img:
        :param long_img:
        :return loss:
        """
        loss_info = {}
        for k, v in self.criterions.items():
            if isinstance(v, L.DistillLossBase):
                loss_array, big_outs = v(small_outs, short_img, long_img)
        
        loss_sum = 0.
        for i in range(len(loss_array)):
            loss_sum += loss_array[i]
        loss_sum *= self.loss_conf['DistillLoss'].weight

        if hasattr(self.loss_conf['DistillLoss'], 'loss_name') and \
             self.loss_conf['DistillLoss'].loss_name is not None:
            loss_name = self.loss_conf['DistillLoss'].loss_name
            assert len(loss_name) == len(loss_array)
        else:
            loss_name = []
            for i in range(len(loss_array)):
                loss_name.append(str(i))

        for i in range(len(loss_name)):
            loss_info[loss_name[i]] = round(loss_array[i].item() * self.loss_conf['DistillLoss'].weight, 5)
        
        return loss_sum, loss_info, big_outs


    def d_loss(self, fake_out, true_out):
        loss_info = {}
        for k, v in self.criterions.items():
            if 'GAN' in k:
                loss_info[k] = v(fake_out, true_out, flag = 'D')

        loss_sum = 0.
        for k, v in loss_info.items():
            loss_sum += self.loss_conf[k].weight * v
            loss_info[k] = round(self.loss_conf[k].weight * v.item(), 5)

        return loss_sum, loss_info


class TrainingModule(P.Parallel, V.VisualBoard):

    def __init__(self, opt, num_gpus:int, rank:int = None, world_size:int = None):
        """
        Base module of Trainer
        :param opt:
        :param num_gpus:
        :param rank:
        :param world_size:
        """
        P.Parallel.__init__(self, num_gpus = num_gpus, rank = rank, world_size = world_size)
        V.VisualBoard.__init__(self, log_path = opt.log_path)

        self.opt = opt

        # cudnn benchmark
        cudnn.benchmark = opt.cudnn_benchmark

        self.save_folder = opt.save_path
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        if not os.path.exists(os.path.join(self.save_folder, "GNet")):
            os.makedirs(os.path.join(self.save_folder, 'GNet'))
        if not os.path.exists(os.path.join(self.save_folder, 'DNet')):
            os.makedirs(os.path.join(self.save_folder, 'DNet'))
        if not os.path.exists(os.path.join(self.save_folder, 'sample')):
            os.makedirs(os.path.join(self.save_folder, 'sample'))

        print("There are %d GPUs used" % self.num_gpus)
        if num_gpus > 0:
            self.opt.Training_config.train_batch_size *= num_gpus
            self.opt.Training_config.val_batch_size *= num_gpus

    def _adjust_learning_rate(self, optim_conf, epoch, iter, optimizer, lr_gd):
        """
        Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        :param optim_conf:
        :param iter:
        :param optimizer:
        :param lr_gd:
        :return:
        """
        if optim_conf.lr_decrease_mode == 'epoch':
            lr = lr_gd * (optim_conf.lr_decrease_factor ** (epoch // optim_conf.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if optim_conf.lr_decrease_mode == 'iter':
            lr = lr_gd * (optim_conf.lr_decrease_factor ** (iter // optim_conf.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def _save_G(self, opt, epoch, iter, len_dataset, net):
        if opt.save_mode == 'epoch':
            save_path = os.path.join(self.save_folder, 'GNet', 'GNet-epoch-%d.pkl' % epoch)
        elif opt.save_mode == 'iter':
            save_path = os.path.join(self.save_folder, 'GNet', 'GNet-iter-%d.pkl' % iter)

        self._save_model(opt, epoch, iter, len_dataset, net, save_path)

    def _save_D(self, opt, epoch, iter, len_dataset, net):
        if opt.save_mode == 'epoch':
            save_path = os.path.join(self.save_folder, 'DNet', 'DNet-epoch-%d.pkl' % epoch)
        elif opt.save_mode == 'iter':
            save_path = os.path.join(self.save_folder, 'DNet', 'DNet-iter-%d.pkl' % iter)

        self._save_model(opt, epoch, iter, len_dataset, net, save_path)

    def _save_model(self, opt, epoch, iter, len_dataset, net, save_path):
        """
        :param opt:
        :param epoch:
        :param iter:
        :param len_dataset:
        :param net:
        :return:
        """
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
            save_state_dict = net.module.state_dict()
        else:
            save_state_dict = net.state_dict()

        if opt.save_mode == 'epoch':
            if epoch % opt.save_by_epoch == 0 and iter % len_dataset == 0:
                torch.save(save_state_dict, save_path)
                print('The trained model is successfully saved in %s' % (save_path))
        elif opt.save_mode == 'iter':
            if iter % opt.save_by_iter == 0:
                torch.save(save_state_dict, save_path)
                print('The trained model is successfully saved in %s' % (save_path))

    def train(self):
        raise NotImplementedError('Not implemented')

    def finish(self):
        self.close()
