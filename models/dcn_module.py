import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from dcn.deform_conv import ModulatedDeformConvPack2 as DCN

class Align_module(nn.Module):

    def __init__(self, channels=32, groups=8):
        super().__init__()

        self.conv_1 = nn.Conv2d(2*channels, channels, 1, 1)
        self.offset_conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.dcnpack = DCN(channels, channels, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True, offset_in_channel=32)
        self.up = nn.ConvTranspose2d(2*channels, channels, 2, 2)
        self.conv_2 = nn.Conv2d(2*channels, channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, short_fea, long_fea, prev_offset=None):
        
        # compute offset and modulation scalars
        comb_fea = torch.cat([short_fea, long_fea], 1)
        offset = self.conv_1(comb_fea)
        offset = self.offset_conv1(offset)
        offset = self.lrelu(offset)
        
        if prev_offset is not None:
            prev_offset = F.interpolate(prev_offset, scale_factor=2, mode='bilinear', align_corners=False)
            comb_offset = torch.cat([offset, prev_offset * 2], dim=1)
            offset = self.offset_conv2(comb_offset)
            offset = self.lrelu(offset)

        offset = self.offset_conv3(offset)
        offset = self.lrelu(offset)

        # deformable conv
        out = self.lrelu(self.dcnpack([long_fea, offset]))

        return out, offset
