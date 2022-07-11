import torch
import torch.nn as nn
from collections import OrderedDict

from ..utils import load_dict
from util.singleton import Singleton
from util.utils import normalize_ImageNet_stats


# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),   # 0   conv1_1
            nn.ReLU(inplace = True),     # 1   relu1_1
            nn.Conv2d(64, 64, 3, 1, 1),  # 2   conv1_2
            nn.ReLU(inplace = True),     # 3   relu1_2
            nn.MaxPool2d(2, 2),          # 4   
            nn.Conv2d(64, 128, 3, 1, 1), # 5   conv2_1
            nn.ReLU(inplace = True),     # 6   relu2_1
            nn.Conv2d(128, 128, 3, 1, 1),# 7   conv2_2
            nn.ReLU(inplace = True),     # 8   relu2_2
            nn.MaxPool2d(2, 2),          # 9
            nn.Conv2d(128, 256, 3, 1, 1),# 10  conv3_1
            nn.ReLU(inplace = True),     # 11  relu3_1
            nn.Conv2d(256, 256, 3, 1, 1),# 12  conv3_2
            nn.ReLU(inplace = True),     # 13  relu3_2
            nn.Conv2d(256, 256, 3, 1, 1),# 14  conv3_3
            nn.MaxPool2d(2, 2),          # 15
            nn.Conv2d(256, 512, 3, 1, 1),# 16  conv4_1
            nn.ReLU(inplace = True),     # 17  relu4_1
            nn.Conv2d(512, 512, 3, 1, 1),# 18  conv4_2
            nn.ReLU(inplace = True),     # 19  relu4_2
            nn.Conv2d(512, 512, 3, 1, 1) # 20  conv4_3
        )

    def forward(self, x):
        x = self.features(x)
        return x


def create_perceptualnet(vgg16_model_path):
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = PerceptualNet()
    # Pre-trained VGG-16
    vgg16 = torch.load(vgg16_model_path)
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
    return perceptualnet


class VGG(nn.Module, metaclass = Singleton):

    """
    A VGG warper, containing cache(FIFO) to store vgg feature.
    """

    def __init__(self, vgg16_model_path):
        super(VGG, self).__init__()
        self.vgg = create_perceptualnet(vgg16_model_path)
        self.cache = OrderedDict()
        self.conv2_2 = []
        self.conv3_2 = []
        self.conv4_2 = []
        self.conv4_3 = []
        for i in range(0, 8):
            self.conv2_2.append(self.vgg.features[i])
        for i in range(8, 13):
            self.conv3_2.append(self.vgg.features[i])
        for i in range(13, 19):
            self.conv4_2.append(self.vgg.features[i])
        for i in range(19, 21):
            self.conv4_3.append(self.vgg.features[i])
        self.conv2_2 = nn.Sequential(*self.conv2_2)
        self.conv3_2 = nn.Sequential(*self.conv3_2)
        self.conv4_2 = nn.Sequential(*self.conv4_2)
        self.conv4_3 = nn.Sequential(*self.conv4_3)

    def forward(self, x):
        id_ = id(x)
        # print(id_)
        # print(self.cache.keys())
        if id_ in self.cache:
            return self.cache[id_]
        else:
            x = normalize_ImageNet_stats(x)
            conv2_2 = self.conv2_2(x)
            conv3_2 = self.conv3_2(conv2_2)
            conv4_2 = self.conv4_2(conv3_2)
            conv4_3 = self.conv4_3(conv4_2)
            self.cache[id_] = [conv2_2, conv3_2, conv4_2, conv4_3]
            return self.cache[id_]

        # x = normalize_ImageNet_stats(x)
        # conv2_2 = self.conv2_2(x)
        # conv3_2 = self.conv3_2(conv2_2)
        # conv4_2 = self.conv4_2(conv3_2)
        # conv4_3 = self.conv4_3(conv4_2)
        # return [conv2_2, conv3_2, conv4_2, conv4_3]
