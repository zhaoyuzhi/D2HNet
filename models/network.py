import torch
import torch.nn as nn

from models.network_module import *
from models import dcn_module

from util.singleton import Singleton


# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ---------------------------------------
#       Base Model for network
# ---------------------------------------
class BaseModel(nn.Module):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt

    @staticmethod
    def load_config_from_json(file):
        pass

    def dump_config_to_json(self, file):
        pass

    def load_ckpt(self, model_path, force_load = False):
        state_dict = torch.load(model_path, map_location = torch.device('cpu'))

        if force_load:
            state_dict_temp = self.state_dict()
            key_temp = set(list(state_dict_temp.keys()))

            for n, p in state_dict.items():
                # temp code
                # if 'res_layer' in n:
                #     n = n.replace('res_layers.', 'res_block_')

                key_temp.remove(n)

                if n in state_dict_temp.keys():
                    if state_dict_temp[n].shape != p.data.shape:
                        print('%s size mismatch, pass!' % n)
                        continue
                    state_dict_temp[n].copy_(p.data)
                else:
                    print('%s not exist, pass!' % n)
            state_dict = state_dict_temp

            if len(key_temp) != 0:
                for k in key_temp:
                    print("param %s not found in state dict!" % k)

        self.load_state_dict(state_dict)
        print("Load checkpoint {} successfully!".format(model_path))


# ----------------------------------------
#                DeblurNet
# ----------------------------------------
class DeblurNet_v2(BaseModel):

    def __init__(self, opt):
        super(DeblurNet_v2, self).__init__(opt)
        # self.downsample_chs = opt.downsample_chs
        self.in_channel = opt.in_channel
        self.out_channel = opt.out_channel
        self.activ = opt.activ
        self.norm = opt.norm
        self.pad_type = opt.pad_type
        self.deblur_res_num = opt.deblur_res_num
        self.deblur_res_num2 = opt.deblur_res_num2
        self.final_activ = opt.final_activ

        if hasattr(opt, 'ngf'):
            self.ngf = opt.ngf
        else:
            self.ngf = 16

        self.build_layers()

    def build_upsample_layer(self, in_channel, out_channel, upsample_level = None):
        if self.opt.upsample_layer == 'pixelshuffle':
            return PixelShuffleAlign(upscale_factor = 2, mode = self.opt.shuffle_mode)
        elif self.opt.upsample_layer == 'bilinear':
            return nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                                 nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1))

    def build_layers(self):
        
        self.dwt = DWT()
        self.idwt = IWT()
        
        self.fusion_conv = Conv2dLayer(self.in_channel * 2 * 4 * 4,
                                       self.ngf * 4 * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                       activation = self.activ, norm = self.norm)

        self.downsample_conv = Conv2dLayer(self.in_channel * 2 * 4,
                                         self.in_channel * 2 * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        
        # deblur resblocks
        for i in range(self.deblur_res_num):
            in_channels = self.ngf * 4 * 4
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            setattr(self, 'deblur_res_block_%d' % i, block)
        
        # upsample layer after deblur resblocks
        self.upsample_conv = Conv2dLayer(self.ngf * 4,
                                         self.ngf * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)

        self.deblur_layer = Conv2dLayer(self.ngf, 3, 3,
                                        stride = 1, padding = 1, pad_type = self.pad_type,
                                        activation = 'none', norm = 'none')

        # deblur resblock2
        for i in range(self.deblur_res_num2):
            in_channels = self.ngf
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            setattr(self, 'deblur_res_block2_%d' % i, block)
        
        if self.opt.final_activ == 'tanh':
            self.final_activ = nn.Tanh()
        
    def forward(self, short_img, long_img):

        sl = torch.cat([short_img, long_img], dim = 1)
        sl = self.dwt(sl)
        sl = self.downsample_conv(sl)
        sl = self.dwt(sl)
        sl = self.fusion_conv(sl)

        deblur_sl = sl

        for i in range(self.deblur_res_num):
            resblock = getattr(self, 'deblur_res_block_%d' % i)
            deblur_sl = resblock(deblur_sl)

        sl = sl + deblur_sl

        sl = self.idwt(sl)
        sl = self.upsample_conv(sl)
        sl = self.idwt(sl)

        deblur_sl = sl

        for i in range(self.deblur_res_num2):
            resblock = getattr(self, 'deblur_res_block2_%d' % i)
            deblur_sl = resblock(deblur_sl)

        sl = deblur_sl + sl

        deblur_sl = self.deblur_layer(sl)

        deblur_out = long_img + deblur_sl
        
        if self.opt.final_activ == 'tanh':
            deblur_out = self.final_activ(deblur_out)

        return deblur_out


# ----------------------------------------
#               DenoiseNet
# ----------------------------------------
class DenoiseNet_v2(BaseModel):

    def __init__(self, opt):
        super(DenoiseNet_v2, self).__init__(opt)
        
        self.in_channel = opt.in_channel
        self.out_channel = opt.out_channel
        self.activ = opt.activ
        self.norm = opt.norm
        self.pad_type = opt.pad_type
        self.denoise_res_num = opt.denoise_res_num
        self.denoise_res_num2 = opt.denoise_res_num2
        self.final_activ = opt.final_activ
        self.groups = opt.groups

        if hasattr(opt, 'ngf'):
            self.ngf = opt.ngf
        else:
            self.ngf = 16

        self.build_layers()

    def build_layers(self):

        self.downsample_short_conv1 = nn.Sequential(
            Conv2dLayer(self.in_channel * 2, self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_short_conv2 = nn.Sequential(
            Conv2dLayer(self.ngf, self.ngf * 2, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_short_conv3 = nn.Sequential(
            Conv2dLayer(self.ngf * 2, self.ngf * 4, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_short_conv4 = nn.Sequential(
            Conv2dLayer(self.ngf * 4, self.ngf * 8, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_short_conv5 = nn.Sequential(
            Conv2dLayer(self.ngf * 8, self.ngf * 16, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_long_conv1 = nn.Sequential(
            Conv2dLayer(self.in_channel, self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_long_conv2 = nn.Sequential(
            Conv2dLayer(self.ngf, self.ngf * 2, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_long_conv3 = nn.Sequential(
            Conv2dLayer(self.ngf * 2, self.ngf * 4, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_long_conv4 = nn.Sequential(
            Conv2dLayer(self.ngf * 4, self.ngf * 8, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        self.downsample_long_conv5 = nn.Sequential(
            Conv2dLayer(self.ngf * 8, self.ngf * 16, 3, stride = 2, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        # decoder level 5
        self.upsample_alignblock5 = dcn_module.Align_module(self.ngf * 16, self.groups)
        self.upsample_comb5 = Conv2dLayer(self.ngf * 32, self.ngf * 16, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_resblock5 = nn.Sequential(
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 16, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )
        
        # decoder level 4
        self.upsample_conv4 = TransposeConv2dLayer(self.ngf * 16,
                                         self.ngf * 8, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        self.upsample_alignblock4 = dcn_module.Align_module(self.ngf * 8, self.groups)
        self.upsample_comb4 = Conv2dLayer(self.ngf * 16, self.ngf * 8, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_resblock4 = nn.Sequential(
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 8, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # decoder level 3
        self.upsample_conv3 = TransposeConv2dLayer(self.ngf * 8 + self.ngf * 8,
                                         self.ngf * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        self.upsample_alignblock3 = dcn_module.Align_module(self.ngf * 4, self.groups)
        self.upsample_comb3 = Conv2dLayer(self.ngf * 8, self.ngf * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_resblock3 = nn.Sequential(
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 4, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # decoder level 2
        self.upsample_conv2 = TransposeConv2dLayer(self.ngf * 4 + self.ngf * 4,
                                         self.ngf * 2, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        self.upsample_alignblock2 = dcn_module.Align_module(self.ngf * 2, self.groups)
        self.upsample_comb2 = Conv2dLayer(self.ngf * 4, self.ngf * 2, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_resblock2 = nn.Sequential(
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm),
            ResBlock(dim = self.ngf * 2, kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        )

        # decoder level 1
        self.upsample_conv1 = TransposeConv2dLayer(self.ngf * 2 + self.ngf * 2,
                                         self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        self.upsample_alignblock1 = dcn_module.Align_module(self.ngf, self.groups)
        self.upsample_comb1 = Conv2dLayer(self.ngf * 2, self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type, activation = self.activ, norm = self.norm)
        self.upsample_conv0 = Conv2dLayer(self.ngf + self.ngf,
                                         self.ngf, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        
        self.short_conv = Conv2dLayer(self.in_channel, self.ngf, 3,
                                      stride = 1, padding = 1, pad_type = self.pad_type,
                                      activation = self.activ, norm = self.norm)
        self.long_conv = Conv2dLayer(self.in_channel, self.ngf, 3,
                                      stride = 1, padding = 1, pad_type = self.pad_type,
                                      activation = self.activ, norm = self.norm)
        self.deblur_out_conv = Conv2dLayer(self.in_channel, self.ngf, 3,
                                      stride = 1, padding = 1, pad_type = self.pad_type,
                                      activation = self.activ, norm = self.norm)

        self.denoise_layer = Conv2dLayer(self.ngf, 3, 3,
                                        stride = 1, padding = 1, pad_type = self.pad_type,
                                        activation = 'none', norm = 'none')
        
        # denoise resblock2
        for i in range(self.denoise_res_num2):
            in_channels = self.ngf
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            setattr(self, 'denoise_res_block2_%d' % i, block)
        
        if self.opt.final_activ == 'tanh':
            self.final_activ = nn.Tanh()
        
    def forward(self, short_img, long_img, deblur_out):

        short_fea1 = torch.cat((short_img, deblur_out), 1)
        short_fea1 = self.downsample_short_conv1(short_fea1)
        short_fea2 = self.downsample_short_conv2(short_fea1)
        short_fea3 = self.downsample_short_conv3(short_fea2)
        short_fea4 = self.downsample_short_conv4(short_fea3)
        short_fea5 = self.downsample_short_conv5(short_fea4)

        long_fea1 = self.downsample_long_conv1(long_img)
        long_fea2 = self.downsample_long_conv2(long_fea1)
        long_fea3 = self.downsample_long_conv3(long_fea2)
        long_fea4 = self.downsample_long_conv4(long_fea3)
        long_fea5 = self.downsample_long_conv5(long_fea4)

        short_fea, offset_5 = self.upsample_alignblock5(short_fea5, long_fea5)
        short_fea = torch.cat((short_fea, short_fea5), 1)   # resolution: 1/16
        short_fea = self.upsample_comb5(short_fea)
        short_fea = self.upsample_resblock5(short_fea)      # resolution: 1/16

        short_fea = self.upsample_conv4(short_fea)          # resolution: 1/8
        short_cut4, offset_4 = self.upsample_alignblock4(short_fea4, long_fea4, offset_5)
        short_cut4 = torch.cat((short_cut4, short_fea4), 1) # resolution: 1/8
        short_cut4 = self.upsample_comb4(short_cut4)
        short_cut4 = self.upsample_resblock4(short_cut4)
        short_fea = torch.cat((short_fea, short_cut4), 1)   # resolution: 1/8

        short_fea = self.upsample_conv3(short_fea)          # resolution: 1/4
        short_cut3, offset_3 = self.upsample_alignblock3(short_fea3, long_fea3, offset_4)
        short_cut3 = torch.cat((short_cut3, short_fea3), 1) # resolution: 1/4
        short_cut3 = self.upsample_comb3(short_cut3)
        short_cut3 = self.upsample_resblock3(short_cut3)
        short_fea = torch.cat((short_fea, short_cut3), 1)   # resolution: 1/4

        short_fea = self.upsample_conv2(short_fea)          # resolution: 1/2
        short_cut2, offset_2 = self.upsample_alignblock2(short_fea2, long_fea2, offset_3)
        short_cut2 = torch.cat((short_cut2, short_fea2), 1) # resolution: 1/2
        short_cut2 = self.upsample_comb2(short_cut2)
        short_cut2 = self.upsample_resblock2(short_cut2)
        short_fea = torch.cat((short_fea, short_cut2), 1)   # resolution: 1/2

        short_fea = self.upsample_conv1(short_fea)          # resolution: 1
        short_cut1, offset_1 = self.upsample_alignblock1(short_fea1, long_fea1, offset_2)
        short_cut1 = torch.cat((short_cut1, short_fea1), 1) # resolution: 1
        short_cut1 = self.upsample_comb1(short_cut1)
        short_fea = torch.cat((short_fea, short_cut1), 1)   # resolution: 1
        short_fea = self.upsample_conv0(short_fea)

        '''
        short_feat = self.short_conv(short_img)
        long_feat = self.long_conv(long_img)
        deblur_out_feat = self.long_conv(deblur_out)
        short_fea = short_fea + short_feat + long_feat + deblur_out_feat
        '''
        sl = short_fea

        for i in range(self.denoise_res_num2):
            resblock = getattr(self, 'denoise_res_block2_%d' % i)
            sl = resblock(sl)

        short_fea = sl + short_fea

        short_fea = self.denoise_layer(short_fea)
        
        denoise_out = deblur_out + short_fea
        
        if self.opt.final_activ == 'tanh':
            denoise_out = self.final_activ(denoise_out)

        return denoise_out


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channel', type = int, default = 3, help = '')
    parser.add_argument('--out_channel', type = int, default = 3, help = '')
    parser.add_argument('--ngf', type = int, default = 64, help = '')
    parser.add_argument('--ngf2', type = int, default = 8, help = '')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = '')
    parser.add_argument('--norm', type = str, default = 'none', help = '')
    parser.add_argument('--pad', type = str, default = 'zero', help = '')

    parser.add_argument('--deblur_res_num', type = int, default = 8, help = '')
    parser.add_argument('--deblur_res_num2', type = int, default = 4, help = '')
    parser.add_argument('--denoise_res_num', type = int, default = 8, help = '')
    parser.add_argument('--denoise_res_num2', type = int, default = 4, help = '')
    parser.add_argument('--groups', type = int, default = 8, help = '')
    
    parser.add_argument('--final_activ', type = str, default = 'none', help = '')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = '')
    parser.add_argument('--upsample_layer', type = str, default = 'pixelshuffle', help = '')
    parser.add_argument('--shuffle_mode', type = str, default = 'caffe', help = '')
    
    opt = parser.parse_args()
    
    a = torch.randn(1, 3, 256, 256).cuda()

    net = DeblurNet_v2(opt).cuda()
    out = net(a, a)

    #net = DenoiseNet_v2(opt).cuda()
    #out = net(a, a, a)

    print(out.shape)
    #save_state_dict = net.state_dict()
    #save_path = 'GNet-epoch-1.pkl'
    #torch.save(save_state_dict, save_path)
