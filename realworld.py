import argparse
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
import math
import os

from util import utils

from models.utils import create_generator_val
from dataloader import dataset
from util.patch_gen import PatchGenerator


def OnePhase_Test(args):
    opt = args.opt

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(args.opt, mode = 'r') as f:
        opt = edict(yaml.load(f))

    generator = create_generator_val(opt.GNet, args.model_path, force_load = True)

    if args.num_gpus >= 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    generator = generator.to(device)

    # Define the dataset
    val_dataset = dataset.SLRGB2RGB_valdataset_singleimage(args.src_path)
    print('The overall number of validation images:', len(val_dataset))

    # Define the dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = 2)

    # forward
    for i, data in enumerate(val_loader):

        # To device
        short_img = data['short_img'].to(device)
        long_img = data['long_img'].to(device)
        short_iso = data['short_iso']
        long_iso = data['long_iso']
        
        # Forward propagation
        with torch.no_grad():
            if args.enable_patch:
                _, _, H, W = short_img.shape 
                patch_size = args.patch_size
                patchGen = PatchGenerator(H, W, patch_size)
                out = torch.zeros_like(short_img)
                deblur_out = torch.zeros_like(short_img)

                for (h, w, top_padding, left_padding, bottom_padding, right_padding) in patchGen.next_patch():
                    short_patch = short_img[:, :, h:h+patch_size, w:w+patch_size]
                    long_patch = long_img[:, :, h:h+patch_size, w:w+patch_size]
                    out_patchs = generator(short_patch, long_patch)

                    if isinstance(out_patchs, list):
                        out_patch = out_patchs[0]
                        deblur_patch = out_patchs[1]
                        deblur_out[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                            deblur_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
                    else:
                        out_patch = out_patchs
                        deblur_out = None
                    
                    out[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                                out_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
            else:
                outs = generator(short_img, long_img)
                if isinstance(outs, list):
                    out = outs[0]
                    deblur_out = outs[1]
                else:
                    out = outs
                    deblur_out = None
        
        short_img = short_img[:, [2, 1, 0], :, :]
        long_img = long_img[:, [2, 1, 0], :, :]
        out = out[:, [2, 1, 0], :, :]
        
        ### Sample data every epoch
        img_list = [short_img, long_img, out]
        name_list = ['inshort_%d' % short_iso, 'inlong_%d' % long_iso, 'pred_' + args.postfix]
        utils.save_sample_png(sample_folder = args.save_path, sample_name = 'val_%d' % (i + 1), img_list = img_list,\
            name_list = name_list, pixel_max_cnt = 255)

        if args.save_deblur and deblur_out is not None:
            deblur_out = deblur_out[:, [2, 1, 0], :, :]
            img_list = [deblur_out]
            name_list = ['deblur' + '_' + args.postfix]
            utils.save_sample_png(sample_folder = args.save_path, sample_name = 'val_%d' % (i+1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)


def TwoPhase_Test(args):
    opt = args.opt

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(args.opt, mode = 'r') as f:
        opt = edict(yaml.load(f))

    if opt.Training_config.phase == 'deblur':
        deblurNet = create_generator_val(opt.DeblurNet, args.model_path, force_load = False)
    elif opt.Training_config.phase == 'denoise':
        denoiseNet = create_generator_val(opt.DenoiseNet, args.model_path, force_load = False)
        deblurNet = create_generator_val(opt.DeblurNet, opt.DeblurNet.finetune_path)
        for param in deblurNet.parameters():
            param.requires_grad = False
    else:
        raise ValueError('phase should be deblur or denoise, but found %s.' % opt.phase)

    if args.num_gpus >= 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deblurNet = deblurNet.to(device)
    if opt.Training_config.phase == 'denoise':
        denoiseNet = denoiseNet.to(device)

    # Define the dataset
    val_dataset = dataset.TP_valdataset_singleimage(args.src_path)
    print('The overall number of validation images:', len(val_dataset))

    # Define the dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = 2)

    # forward
    for i, data in enumerate(val_loader):

        # To device
        short_img = data['in_short_img'].to(device)
        long_img = data['in_long_img'].to(device)
        down_short_img = data['down_short_img'].to(device)
        down_long_img = data['down_long_img'].to(device)
        short_iso = data['short_iso']
        long_iso = data['long_iso']
        tag = 0
        
        # Forward propagation
        with torch.no_grad():

            deblur_out = deblurNet(down_short_img, down_long_img)
            deblur_out = F.interpolate(deblur_out, size = (short_img.shape[2], short_img.shape[3]), mode = 'bilinear', align_corners = False)
            
            img_list = []
            name_list = []

            if opt.Training_config.phase == 'denoise':

                if args.enable_patch:
                    _, _, H, W = short_img.shape
                    patch_size = args.patch_size
                    patchGen = PatchGenerator(H, W, patch_size)
                    out = torch.zeros_like(short_img)
                    attn = torch.zeros_like(short_img)

                    for (h, w, top_padding, left_padding, bottom_padding, right_padding) in patchGen.next_patch():
                        short_patch = short_img[:, :, h:h+patch_size, w:w+patch_size]
                        long_patch = long_img[:, :, h:h+patch_size, w:w+patch_size]
                        deblur_patch = deblur_out[:, :, h:h+patch_size, w:w+patch_size]
                        out_patch = denoiseNet(short_patch, long_patch, deblur_patch)
                        
                        if isinstance(out_patch, list):
                            tag = 1
                            attn_patch = out_patch[1]
                            attn_patch = torch.cat((attn_patch, attn_patch, attn_patch), 1)
                            out_patch = out_patch[0]
                        else:
                            out_patch = out_patch
                        
                        out[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                            out_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
                        if tag == 1:
                            attn[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                                attn_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
                                
                else:
                    out = denoiseNet(short_img, long_img, deblur_out)
        
                    if isinstance(out, list):
                        tag = 1
                        attn = out[1]
                        attn = torch.cat((attn, attn, attn), 1)
                        out = out[0]
                    else:
                        out = out

                out = out[:, [2, 1, 0], :, :]
                
            short_img = short_img[:, [2, 1, 0], :, :]
            long_img = long_img[:, [2, 1, 0], :, :]
            deblur_out = deblur_out[:, [2, 1, 0], :, :]
                
        if args.save_deblur and deblur_out is not None:
            img_list += [short_img]
            name_list += ['short']
            img_list += [long_img]
            name_list += ['long']
            img_list += [deblur_out]
            name_list += ['firstout']
            if opt.Training_config.phase == 'denoise':
                img_list += [out]
                name_list += ['secondout']
                if tag == 1:
                    img_list += [attn]
                    name_list += ['attn']
            utils.save_sample_png(sample_folder = args.save_path, sample_name = 'val_%d' % (i+1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type = str, \
        default = './options/tp_denoisenet_v2_002.yaml', \
            help = 'Path to option YAML file.')
    parser.add_argument('--model_path', type = str, \
        default = './snapshot/tp_denoisenet_v2_002/GNet/GNet-epoch-149.pkl', \
            help = 'Model path to load.')
    parser.add_argument('--src_path', type = str, \
        default = './data/Xiaomi_Mi_Note_10_photos', \
        #default = '/media/zyz/Seagate Backup Plus Drive/D2HNet dataset/mobile_phone/Xiaomi_Mi_Note_10_photos', \
            help = 'Image path to read.')
    parser.add_argument('--save_path', type = str, \
        default = './results_real_photo/tp_denoisenet_v2_002', \
            help = 'Path to save images.')
    parser.add_argument('--num_gpus', type = int, default = 1, help = 'GPU number, 0 means cpu is used.')
    parser.add_argument('--enable_patch', type = bool, default = True, help = 'enable patch process.')
    parser.add_argument('--patch_size', type = int, default = 1024, help = 'patch size.')
    parser.add_argument('--save_deblur', type = bool, default = True)
    parser.add_argument('--postfix', type = str, default = '')
    args = parser.parse_args()

    TwoPhase_Test(args)
    