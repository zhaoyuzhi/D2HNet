import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

import numpy as np


# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                 dilation = 1, pad_type = 'zero',
                 activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        self.layers = []

        # Initialize the padding scheme
        p = padding
        if p > 0:
            if pad_type == 'reflect':
                self.layers += [nn.ReflectionPad2d(padding)]
            elif pad_type == 'replicate':
                self.layers += [nn.ReplicationPad2d(padding)]
            elif pad_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding type %s is not supported.' % pad_type)

        # Initialize the convolution layers
        if sn:
            self.layers += [SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = p, dilation = dilation))]
        else:
            self.layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = p, dilation = dilation)]

        # Initialize the normalization type
        if norm == 'bn':
            self.layers += [nn.BatchNorm2d(out_channels)]
        elif norm == 'in':
            self.layers += [nn.InstanceNorm2d(out_channels)]
        elif norm == 'ln':
            self.layers += [LayerNorm(out_channels)]
        elif norm == 'none':
            pass
        else:
            raise NotImplementedError('norm layer %s is not supported.' % norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.layers += [nn.ReLU(inplace = True)]
        elif activation == 'lrelu':
            self.layers += [nn.LeakyReLU(0.2, inplace = True)]
        elif activation == 'prelu':
            self.layers += [nn.PReLU()]
        elif activation == 'selu':
            self.layers += [nn.SELU(inplace = True)]
        elif activation == 'tanh':
            self.layers += [nn.Tanh()]
        elif activation == 'sigmoid':
            self.layers += [nn.Sigmoid()]
        elif activation == 'none':
            pass
        else:
            raise NotImplementedError('activation %s is not supported.' % activation)

        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x


# ----------------------------------------
#                ResBlock
# ----------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero',
                 activation = 'lrelu', norm = 'none', sn = False):
        super(ResBlock, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(dim, dim, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn),
            Conv2dLayer(dim, dim, kernel_size, stride, padding, dilation, pad_type, activation = 'none', norm = norm, sn = sn)
        )
        
    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out


# ----------------------------------------
#            ConvLSTM2d Block
# ----------------------------------------
class ConvLSTM2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name = 'weight', power_iterations = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# ----------------------------------------
#              PixelShuffle
# ----------------------------------------
class PixelShuffleAlign(nn.Module):
    def __init__(self, upscale_factor: int = 2, mode: str = 'caffe'):
        """
        :param upscale_factor: upsample scale
        :param mode: caffe, pytorch
        """
        super(PixelShuffleAlign, self).__init__()
        self.upscale_factor = upscale_factor
        self.mode = mode

    def forward(self, x):
        # assert len(x.size()) == 4, "Received input tensor shape is {}".format(
        #     x.size())
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = C // (self.upscale_factor ** 2)
        h, w = H * self.upscale_factor, W * self.upscale_factor

        if self.mode == 'caffe':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, self.upscale_factor,
                          self.upscale_factor, c, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2)
        elif self.mode == 'pytorch':
            # (N, C, H, W) => (N, c, r, r, H, W)
            x = x.reshape(-1, c, self.upscale_factor,
                          self.upscale_factor, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x


class PixelUnShuffleAlign(nn.Module):

    def __init__(self, downscale_factor: int = 2, mode: str = 'caffe'):
        """
        :param downscale_factor: downsample scale
        :param mode: caffe, pytorch
        """
        super(PixelUnShuffleAlign, self).__init__()
        self.dsf = downscale_factor
        self.mode = mode

    def forward(self, x):
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = int(C * (self.dsf ** 2))
        h, w = H // self.dsf, W // self.dsf

        x = x.reshape(-1, C, h, self.dsf, w, self.dsf)
        if self.mode == 'caffe':
            x = x.permute(0, 3, 5, 1, 2, 4)
        elif self.mode == 'pytorch':
            x = x.permute(0, 1, 3, 5, 2, 4)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x


# ----------------------------------------
#                DWT / IDWT
# ----------------------------------------
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# ----------------------------------------
#              Self Attention
# ----------------------------------------
class AttnModule(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, efficient = False, pool = False):
        super(AttnModule, self).__init__()
        self.chanel_in = in_dim
        # if efficient is True, use adaptive pool
        self.efficient = efficient
        self.pool = pool

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim = -1)  #

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                y : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, _, W, H = x.size()
        if self.efficient:
            if W > 64 or H > 64:
                x = F.adaptive_max_pool2d(x, 64)
                y = F.adaptive_max_pool2d(y, 64)

        if self.pool:
            x = F.max_pool2d(x, 3, 2, 1)
            y = F.max_pool2d(y, 3, 2, 1)

        m_batchsize, C, width, height = y.size()
        proj_query = self.query_conv(y).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        if self.efficient or self.pool:
            out = F.interpolate(out, size = (W, H), mode = 'bilinear')

        return out, attention


# ----------------------------------------
#                GradLayer
# ----------------------------------------
class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim = 1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding = 1)
        x_h = F.conv2d(x, self.weight_h, padding = 1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class BlurLayer(nn.Module):
    """Implements the blur layer used in StyleGAN."""

    def __init__(self, channels, kernel = (1, 2, 1), normalize = True, flip = False):
        super().__init__()
        kernel = np.array(kernel, dtype = np.float32).reshape(1, 3)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        if flip:
            kernel = kernel[::-1, ::-1]
        kernel = kernel.reshape(3, 3, 1, 1)
        kernel = np.tile(kernel, [1, 1, channels, 1])
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride = 1, padding = 1, groups = self.channels)
