import argparse
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
import numpy as np
import cv2

from util import utils
from util.patch_gen import PatchGenerator

from models.utils import create_generator_val
from dataloader import dataset

class ValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.val_path = opt.val_path + str(opt.val_res) + 'p'
        self.imglist = self.get_heads(self.val_path)

    def get_heads(self, path):
        # read a folder, return the image name
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                head = filespath.split('_')[0] + '_' + filespath.split('_')[1]
                if os.path.join(root.split('/')[-1], head) not in ret:
                    ret.append(os.path.join(root.split('/')[-1], head))
        return ret

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        long_img_path = os.path.join(self.val_path, imgname + '_long8.png')
        short_img_path = os.path.join(self.val_path, imgname + '_short.png')
        RGBout_img_path = os.path.join(self.opt.val_sharp_path, imgname + self.opt.postfix + '.png')
        save_path = os.path.join(self.opt.save_path, imgname + '_pred.png')

        long_img = cv2.imread(long_img_path)
        short_img = cv2.imread(short_img_path)
        RGBout_img = cv2.imread(RGBout_img_path)
        RGBout_img = cv2.resize(RGBout_img, (int(2560 / 1440 * self.opt.val_res), self.opt.val_res))

        long_img = cv2.cvtColor(long_img, cv2.COLOR_BGR2RGB)
        short_img = cv2.cvtColor(short_img, cv2.COLOR_BGR2RGB)
        RGBout_img = cv2.cvtColor(RGBout_img, cv2.COLOR_BGR2RGB)
        down_long_img = cv2.resize(long_img, (short_img.shape[1] // 2, short_img.shape[0] // 2), \
            interpolation = cv2.INTER_AREA)
        down_short_img = cv2.resize(short_img, (short_img.shape[1] // 2, short_img.shape[0] // 2), \
            interpolation = cv2.INTER_AREA)

        long_img = long_img.astype(np.float) / 255.
        short_img = short_img.astype(np.float) / 255.
        RGBout_img = RGBout_img.astype(np.float) / 255.
        down_long_img = down_long_img.astype(np.float) / 255.
        down_short_img = down_short_img.astype(np.float) / 255.

        long_img = torch.from_numpy(long_img).float().permute(2, 0, 1)
        short_img = torch.from_numpy(short_img).float().permute(2, 0, 1)
        RGBout_img = torch.from_numpy(RGBout_img).float().permute(2, 0, 1)
        down_long_img = torch.from_numpy(down_long_img).float().permute(2, 0, 1)
        down_short_img = torch.from_numpy(down_short_img).float().permute(2, 0, 1)
        
        sample = {'down_short_img': down_short_img,
                  'down_long_img': down_long_img,
                  'long_img': long_img,
                  'short_img': short_img,
                  'RGBout_img': RGBout_img,
                  'save_path': save_path}

        return sample


def TwoPhase_Val(args):

    opt = args.opt

    if args.save_deblur:
        utils.check_path(os.path.join(args.save_path, 'day'))
        utils.check_path(os.path.join(args.save_path, 'night'))

    with open(args.opt, mode = 'r') as f:
        opt = edict(yaml.load(f))

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    
    # Define the network
    if opt.Training_config.phase == 'deblur':
        deblurNet = create_generator_val(opt.DeblurNet, args.model_path, force_load=False)
    elif opt.Training_config.phase == 'denoise':
        denoiseNet = create_generator_val(opt.DenoiseNet, args.model_path, force_load=True)
        deblurNet = create_generator_val(opt.DeblurNet, opt.DeblurNet.finetune_path)
        for param in deblurNet.parameters():
            param.requires_grad = False

    if args.num_gpus >= 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deblurNet = deblurNet.to(device)
    if opt.Training_config.phase == 'denoise':
        denoiseNet = denoiseNet.to(device)

    # Define the dataset
    val_dataset = ValDataset(args)
    print('The overall number of validation images:', len(val_dataset))

    # Define the dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Initialize
    utils.check_path(args.save_path)

    # forward
    val_PSNR = 0
    val_SSIM = 0

    for i, data in enumerate(val_loader):

        # To device
        short_img = data['short_img'].to(device)
        long_img = data['long_img'].to(device)
        down_short_img = data['down_short_img'].to(device)
        down_long_img = data['down_long_img'].to(device)
        RGBout_img = data['RGBout_img'].to(device)
        save_path = data['save_path'][0]
        
        # Forward propagation
        with torch.no_grad():
            
            deblur_out = deblurNet(down_short_img, down_long_img)
            deblur_out_residual = deblur_out - down_long_img
            deblur_out = F.interpolate(deblur_out, size=(short_img.shape[2], short_img.shape[3]), mode='bilinear', align_corners=False)

            if opt.Training_config.phase == 'denoise':

                if args.enable_patch:
                    _, _, H, W = short_img.shape 
                    patch_size = args.patch_size
                    patchGen = PatchGenerator(H, W, patch_size)
                    out = torch.zeros_like(short_img)

                    for (h, w, top_padding, left_padding, bottom_padding, right_padding) in patchGen.next_patch():
                        short_patch = short_img[:, :, h:h+patch_size, w:w+patch_size]
                        long_patch = long_img[:, :, h:h+patch_size, w:w+patch_size]
                        deblur_patch = deblur_out[:, :, h:h+patch_size, w:w+patch_size]
                        out_patch = denoiseNet(short_patch, long_patch, deblur_patch)

                        if isinstance(out_patch, list):
                            out_patch = out_patch[0]
                        
                        out[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                            out_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
                else:
                    out = denoiseNet(short_img, long_img, deblur_out)

                    if isinstance(out, list):
                        out = out[0]
                
                out_residual = out - deblur_out
            
            elif opt.Training_config.phase == 'deblur':
                out = deblur_out
        
        # Save the image (BCHW -> HWC)
        if args.save_deblur:
            if opt.Training_config.phase == 'deblur':
                save_img = torch.clamp(deblur_out, 0, 1)
                save_img = save_img[0, :, :, :].permute(1, 2, 0).cpu().data.numpy()
                save_img = (save_img * 255).astype(np.uint8)
                save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path[:-8] + 'deblur.png', save_img)

                # if args.save_residual:
                #     save_img = deblur_out_residual.cpu().data.numpy()
                #     np.save(save_path[:-8] + 'deblur_residual.npy', save_img)
            
            if opt.Training_config.phase == 'denoise':
                save_img = torch.clamp(out, 0, 1)
                save_img = save_img[0, :, :, :].permute(1, 2, 0).cpu().data.numpy()
                save_img = (save_img * 255).astype(np.uint8)
                save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path, save_img)
        
                # if args.save_residual:
                #     save_img = out_residual.cpu().data.numpy()
                #     np.save(save_path[:-8] + 'pred_residual.npy', save_img)
            
        # PSNR
        # print('The %d-th image PSNR %.4f' % (i, val_PSNR_this))
        this_PSNR = utils.psnr(out, RGBout_img, 1) * short_img.shape[0]
        val_PSNR += this_PSNR
        this_SSIM = utils.ssim(out, RGBout_img) * short_img.shape[0]
        val_SSIM += this_SSIM
        print('The %d-th image %s: PSNR: %.5f, SSIM: %.5f' % (i + 1, save_path, this_PSNR, this_SSIM))

    val_PSNR = val_PSNR / len(val_dataset)
    val_SSIM = val_SSIM / len(val_dataset)
    print('The average of %s: PSNR: %.5f, average SSIM: %.5f' % (args.opt, val_PSNR, val_SSIM))


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type = str, \
        default = './options/tp_denoisenet_v2_002.yaml', \
            help='Path to option YAML file.')
    parser.add_argument('--model_path', type = str, \
        default = './snapshot/tp_denoisenet_v2_002/GNet/GNet-official.pkl', \
            help = 'Model path to load.')
    parser.add_argument('--val_path', type = str, \
        default = './data/slrgb2rgb_simulated_mobile_phone/val_no_overlap_noisy_', \
            help = 'Image path to read.')
    parser.add_argument('--val_res', type = int, default = 1440, help = 'validation resolution')
    parser.add_argument('--val_sharp_path', type = str, \
        default = './data/slrgb2rgb_simulated_mobile_phone_sharp/val_no_overlap', \
            help = 'Image path to ground truth.')
    parser.add_argument('--save_path', type = str, \
        default = './results_validation/tp_denoisenet_v2_002', \
            help = 'Path to save images.')
    parser.add_argument('--num_gpus', type = int, default = 1, help = 'GPU number, 0 means cpu is used.')
    parser.add_argument('--down_img_size', type = int, default = 1024, help = 'down_img_size')
    parser.add_argument('--enable_patch', type = bool, default = False, help = 'enable patch process, please set True if out of memory.')
    parser.add_argument('--patch_size', type = int, default = 1024, help = 'patch size.')
    parser.add_argument('--save_deblur', type = bool, default = False)
    parser.add_argument('--save_residual', type = bool, default = False)
    parser.add_argument('--postfix', type = str, default = '_short', help = 'suffix of ground truth images')
    args = parser.parse_args()

    TwoPhase_Val(args)
