from tensorboardX import SummaryWriter

import torchvision.utils as t_utils
import torch
import os
import os.path as osp


class VisualBoard:

    def __init__(self, log_path):
        if not osp.exists(log_path):
            os.makedirs(log_path)

        self.writer = SummaryWriter(log_path)

    def visual_image(self, tag, images, iters, normalize=False):
        image_show = t_utils.make_grid(torch.cat(images, dim=0),
                                       nrow=images[0].size()[0],
                                       normalize=normalize,
                                       range=(0, 1))
        self.writer.add_image(tag, image_show, iters)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_scalar(self, tag, scalar, global_step=None):
        self.writer.add_scalar(tag, scalar, global_step)

    def close(self):
        self.writer.close()
