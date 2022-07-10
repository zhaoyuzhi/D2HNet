import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from util import utils
from dataloader import dataset
from models.loss import ColorLoss, VGGLoss
import models.loss as L
from models.optimizer.adamw import AdamW
from models.TrainingModule import TrainingModule, LossManager, pack_network_output, pack_gt_data
from models.utils import create_generator, create_discriminator, create_generator_val


class Trainer(TrainingModule):

    def __init__(self, opt, num_gpus, rank = None, world_size = None):
        super(Trainer, self).__init__(opt = opt,
                                      num_gpus = num_gpus,
                                      rank = rank,
                                      world_size = world_size)

        self.Training_config = self.opt.Training_config
        self.optim_config = self.opt.Optimizer

        self.G = create_generator(opt.GNet)
        self.D = create_discriminator(opt.DNet)
        # to support multi gpu or distributed training
        self.G = self.wrapper(self.G)
        self.D = self.wrapper(self.D)

        self.LM = LossManager(self.opt.Loss, num_gpus = num_gpus)

        self._init_dataloader()
        self._init_optim()

    def _init_dataloader(self):
        # Define the dataset
        train_dataset = dataset.SLRGB2RGB_dataset(self.opt.Dataset, 'train')
        val_dataset = dataset.SLRGB2RGB_dataset(self.opt.Dataset, 'val')
        print('The overall number of training images:', len(train_dataset))
        print('The overall number of validation images:', len(val_dataset))

        # Define the dataloader
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.Training_config.train_batch_size, shuffle = True, num_workers = self.Training_config.num_workers, pin_memory = True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.Training_config.val_batch_size, shuffle = False, num_workers = self.Training_config.num_workers, pin_memory = True)

    def _init_optim(self):
        if self.optim_config.name == "Adam":
            self.optim_G = torch.optim.Adam(self.G.parameters(), lr = self.optim_config.args.lr_g, betas = (self.optim_config.args.b1, self.optim_config.args.b2), weight_decay = self.optim_config.args.weight_decay)
            self.optim_D = torch.optim.Adam(self.D.parameters(), lr = self.optim_config.args.lr_d, betas = (self.optim_config.args.b1, self.optim_config.args.b2), weight_decay = self.optim_config.args.weight_decay)
        elif self.optim_config.name == "SGD":
            self.optim_G = torch.optim.SGD(self.G.parameters(), lr = self.optim_config.args.lr_g)
            self.optim_D = torch.optim.SGD(self.D.parameters(), lr = self.optim_config.args.lr_d)

    def train(self):
        # Count start time
        iters_done = 0
        # self._validate(1)

        for epoch in range(self.Training_config.start_idx + 1, self.Training_config.epochs):
            print('epoch ', epoch)
            # Record learning rate
            for param_group in self.optim_G.param_groups:
                self.add_scalar('lr', param_group['lr'], epoch)

            for i, data in enumerate(self.train_loader):

                print(i, self.device)
                short_img = data['short_img'].to(self.device)
                long_img = data['long_img'].to(self.device)
                RGBout_img = data['RGBout_img'].to(self.device)
                gt_long_img = data['gt_long_img'].to(self.device)

                # process patch
                if len(short_img.shape) == 5:
                    _, _, C, H, W = short_img.shape
                    short_img = short_img.view(-1, C, H, W)
                    long_img = long_img.view(-1, C, H, W)
                    RGBout_img = RGBout_img.view(-1, C, H, W)
                    gt_long_img = gt_long_img.view(-1, C, H, W)

                outs = self.G(short_img, long_img)

                if isinstance(outs, list):
                    out = outs[0]
                else:
                    out = outs
                
                # ========= pack gt data ==========
                gt_dict = pack_gt_data([RGBout_img, gt_long_img])
                
                # ========= pack output ==========
                outputs = pack_network_output(outs, self.opt.GNet.name)

                # ========= train G ============
                if self.LM.train_GAN:
                    fake_scalar = self.D(short_img, long_img, out)
                else:
                    fake_scalar = None
                G_loss, G_loss_info = self.LM(outputs, gt_dict, fake_scalar)

                if self.LM.distill_training:
                    distill_loss, distill_info, big_outs = self.LM.distill_loss(outs, short_img, long_img)
                    G_loss += distill_loss
                    G_loss_info.update(distill_info)

                self.optim_G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

                # ========== train D =============
                if self.LM.train_GAN:
                    fake_scalar_d = self.D(short_img, long_img, out.detach())
                    true_scalar_d = self.D(short_img, long_img, RGBout_img)
                    D_loss, D_loss_info = self.LM.d_loss(fake_scalar_d, true_scalar_d)
                    self.optim_D.zero_grad()
                    D_loss.backward()
                    self.optim_D.step()
                else:
                    D_loss, D_loss_info = 0., {}

                # record loss
                if iters_done % self.Training_config.show_loss_iter == 0:
                    self.add_scalars(main_tag = 'G_loss', tag_scalar_dict = G_loss_info, global_step = iters_done)
                    self.add_scalars(main_tag = 'D_loss', tag_scalar_dict = D_loss_info, global_step = iters_done)
                if iters_done % self.Training_config.show_img_iter == 0:
                    vis_imgs = [short_img, long_img, RGBout_img, gt_long_img]
                    if isinstance(outs, list):
                        for j in range(len(outs)):
                            if outs[j] is not None and outs[j].size()[1] == 3:
                                vis_imgs.append(outs[j].clamp(0.0, 1.0))
                    else:
                        vis_imgs.append(outs)

                    self.visual_image('train_image', vis_imgs, iters_done)

                    # show gradient image
                    if isinstance(outs, list):
                        vis_imgs = []
                        for j in range(len(outs)):
                            if outs[j] is not None and outs[j].size()[1] == 1:
                                vis_imgs.append(outs[j].clamp(0.0, 1.0))
                        
                        if len(vis_imgs) > 0:
                            self.visual_image('train_feat', vis_imgs, iters_done)
                    # show large model output
                    if self.LM.distill_training:
                        vis_imgs = []
                        vis_feats = []
                        for j in range(len(big_outs)):
                            if big_outs[j] is not None: 
                                if big_outs[j].size()[1] == 3:
                                    vis_imgs.append(big_outs[j].clamp(0.0, 1.0))
                                elif big_outs[j].size()[1] == 1:
                                    vis_feats.append(big_outs[j].clamp(0.0, 1.0))
                        
                        if len(vis_imgs) > 0:
                            self.visual_image('distill_image', vis_imgs, iters_done)
                        if len(vis_feats) > 0:
                            self.visual_image('distill_feat', vis_feats, iters_done)
                            

                self._save_G(self.opt, epoch, iters_done, len(self.train_loader), self.G)
                if self.LM.train_GAN:
                    self._save_D(self.opt, epoch, iters_done, len(self.train_loader), self.D)

                self._adjust_learning_rate(self.optim_config, (epoch + 1), iters_done, self.optim_G, self.opt.Optimizer.args.lr_g)
                if self.LM.train_GAN:
                    self._adjust_learning_rate(self.optim_config, (epoch + 1), iters_done, self.optim_D, self.opt.Optimizer.args.lr_d)

                iters_done += 1

            if epoch % 1 == 0:
                img_list = [short_img, long_img, out, RGBout_img]
                name_list = ['inshort', 'inlong', 'pred', 'gt']
                utils.save_sample_png(sample_folder = os.path.join(self.save_folder, 'sample'), sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

            # if epoch % 5 == 0:
            self._validate(epoch)

    def _validate(self, epoch):
        self.G.eval()
        val_PSNR = 0
        val_SSIM = 0
        val_day_PSNR, val_night_PSNR = 0, 0
        val_day_SSIM, val_night_SSIM = 0, 0
        num_of_val_image = 0
        num_of_val_day, num_of_val_night = 0, 0

        for j, dataset in enumerate(self.val_loader):

            short_img = dataset['short_img'].to(self.device)
            long_img = dataset['long_img'].to(self.device)
            RGBout_img = dataset['RGBout_img'].to(self.device)
            short_paths = dataset['in_short_path']

            # process patch
            if len(short_img.shape) == 5:
                _, _, C, H, W = short_img.shape
                short_img = short_img.view(-1, C, H, W)
                long_img = long_img.view(-1, C, H, W)
                RGBout_img = RGBout_img.view(-1, C, H, W)

            out = self.G(short_img, long_img)

            if isinstance(out, list):
                out = out[0]
                # deblur_out = out[1]
                # deblur_out = deblur_out.clamp(0.0, 1.0)
            else:
                # deblur_out = None
                pass

            out = out.clamp(0.0, 1.0)

            num_of_val_image += short_img.shape[0]
            val_PSNR += utils.psnr(out, RGBout_img, 1) * short_img.shape[0]
            val_SSIM += utils.ssim(out, RGBout_img) * short_img.shape[0]

            # calculate psnr, ssim for day and night data
            day_outs, day_gts = [], []
            night_outs, night_gts = [], []
            for idx in range(len(short_paths)):
                if 'day' in short_paths[idx]:
                    day_outs.append(out[idx].unsqueeze(dim = 0))
                    day_gts.append(RGBout_img[idx].unsqueeze(dim = 0))
                elif 'night' in short_paths[idx]:
                    night_outs.append(out[idx].unsqueeze(dim = 0))
                    night_gts.append(RGBout_img[idx].unsqueeze(dim = 0))

            if len(day_outs) >= 1:
                day_outs = torch.cat(day_outs, dim = 0)
                day_gts = torch.cat(day_gts, dim = 0)
                val_day_PSNR += utils.psnr(day_outs, day_gts, 1) * day_outs.shape[0]
                val_day_SSIM += utils.ssim(day_outs, day_gts) * day_outs.shape[0]
                num_of_val_day += day_outs.shape[0]

            if len(night_outs) >= 1:
                night_outs = torch.cat(night_outs, dim = 0)
                night_gts = torch.cat(night_gts, dim = 0)
                val_night_PSNR += utils.psnr(night_outs, night_gts, 1) * night_outs.shape[0]
                val_night_SSIM += utils.ssim(night_outs, night_gts) * night_outs.shape[0]
                num_of_val_night += night_outs.shape[0]


            if j % 10 == 0:
                print('val: %d | epoch: %d' % (j, epoch))

        val_PSNR = val_PSNR / num_of_val_image
        val_SSIM = val_SSIM / num_of_val_image


        val_day_PSNR = val_day_PSNR / num_of_val_day
        val_day_SSIM = val_day_SSIM / num_of_val_day
        val_night_PSNR = val_night_PSNR / num_of_val_night
        val_night_SSIM = val_night_SSIM / num_of_val_night

        self.add_scalar('val_PSNR', val_PSNR, global_step = epoch)
        self.add_scalar('val_SSIM', val_SSIM, global_step = epoch)
        self.add_scalar('val_day_PSNR', val_day_PSNR, global_step = epoch)
        self.add_scalar('val_day_SSIM', val_day_SSIM, global_step = epoch)
        self.add_scalar('val_night_PSNR', val_night_PSNR, global_step = epoch)
        self.add_scalar('val_night_SSIM', val_night_SSIM, global_step = epoch)

        self.G.train()

        print('val: epoch: %d, psnr: %.3f, ssim: %.3f' % (epoch, val_PSNR, val_SSIM))


class TwoPhaseTrainer(TrainingModule):

    def __init__(self, opt, num_gpus, rank = None, world_size = None):
        super(TwoPhaseTrainer, self).__init__(opt = opt,
                                              num_gpus = num_gpus,
                                              rank = rank,
                                              world_size = world_size)

        self.Training_config = self.opt.Training_config
        self.optim_config = self.opt.Optimizer

        if self.Training_config.phase == 'deblur':
            self.G = create_generator(opt.DeblurNet)
        elif self.Training_config.phase == 'denoise':
            self.G = create_generator(opt.DenoiseNet)
            if opt.DeblurNet.finetune_path:
                self.deblurNet = create_generator_val(opt.DeblurNet, opt.DeblurNet.finetune_path)
            else:
                self.deblurNet = create_generator_val(opt.DeblurNet)
            for param in self.deblurNet.parameters():
                param.requires_grad = False
            self.deblurNet = self.wrapper(self.deblurNet)
        else:
            raise ValueError('phase should be deblur or denoise, but found %s.' % opt.phase)


        # to support multi gpu or distributed training
        self.G = self.wrapper(self.G)

        self.LM = LossManager(self.opt.Loss, num_gpus = num_gpus)

        self._init_dataloader()
        self._init_optim()

    def _init_dataloader(self):
        # Define the dataset
        train_dataset = dataset.TP_dataset_v1(self.opt.Dataset, 'train')
        val_dataset = dataset.TP_dataset_v1(self.opt.Dataset, 'val')
        print('The overall number of training images:', len(train_dataset))
        print('The overall number of validation images:', len(val_dataset))

        # Define the dataloader
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.Training_config.train_batch_size, shuffle = True, num_workers = self.Training_config.num_workers, pin_memory = True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.Training_config.val_batch_size, shuffle = False, num_workers = self.Training_config.num_workers, pin_memory = True)

    def _init_optim(self):
        if self.optim_config.name == "Adam":
            self.optim_G = torch.optim.Adam(self.G.parameters(), lr = self.optim_config.args.lr_g, betas = (self.optim_config.args.b1, self.optim_config.args.b2), weight_decay = self.optim_config.args.weight_decay)
            self.optim_D = torch.optim.Adam(self.D.parameters(), lr = self.optim_config.args.lr_d, betas = (self.optim_config.args.b1, self.optim_config.args.b2), weight_decay = self.optim_config.args.weight_decay)
        elif self.optim_config.name == "SGD":
            self.optim_G = torch.optim.SGD(self.G.parameters(), lr = self.optim_config.args.lr_g)
            self.optim_D = torch.optim.SGD(self.D.parameters(), lr = self.optim_config.args.lr_d)
        elif self.optim_config.name == "Adamw":
            self.optim_G = AdamW(self.G.parameters(), lr = self.optim_config.args.lr_g, betas = (self.optim_config.args.b1, self.optim_config.args.b2), weight_decay = self.optim_config.args.weight_decay)
        

    def train(self):
        # Count start time
        iters_done = 0

        for epoch in range(self.Training_config.start_idx + 1, self.Training_config.epochs):
            print('epoch', epoch)
            # self._validate(epoch)

            for param_group in self.optim_G.param_groups:
                self.add_scalar('lr', param_group['lr'], epoch)
            
            for i, data in enumerate(self.train_loader):
                print(i, self.device)

                # ========= Get data ==================
                short_img = data['short_img'].to(self.device)
                long_img = data['long_img'].to(self.device)
                RGBout_img = data['RGBout_img'].to(self.device)
                gt_long_img = data['gt_long_img'].to(self.device)

                if len(short_img.shape) == 5:
                    _, _, C, H, W = short_img.shape
                    short_img = short_img.view(-1, C, H, W)
                    long_img = long_img.view(-1, C, H, W)
                    RGBout_img = RGBout_img.view(-1, C, H, W)
                    gt_long_img = gt_long_img.view(-1, C, H, W)

                down_short_img, down_long_img, down_out_img, down_gtlong_img = \
                    self.train_loader.dataset.downsample_tensors([short_img, long_img, RGBout_img, gt_long_img])
                
                # =========== forward ==================
                if self.Training_config.phase == 'deblur':
                    outs = self.G(down_short_img, down_long_img)
                elif self.Training_config.phase == 'denoise':
                    deblur_out = self.deblurNet(down_short_img, down_long_img).detach()
                    deblur_out = F.upsample(deblur_out, size=(short_img.shape[2], short_img.shape[3]), mode='bilinear', align_corners=False)

                    short_img, long_img, RGBout_img, deblur_out = \
                        self.train_loader.dataset.crop_tensor_patch([short_img, long_img, RGBout_img, deblur_out])
                    deblur_out = self.train_loader.dataset.augment_tensor_patch(deblur_out, RGBout_img)
                    deblur_out = deblur_out.clamp(0.0, 1.0)

                    outs = self.G(short_img, long_img, deblur_out)
                
                # ========== pack gt data =================
                if self.Training_config.phase == 'deblur':
                    gt_dict = pack_gt_data([down_out_img, down_gtlong_img])
                elif self.Training_config.phase == 'denoise':
                    gt_dict = pack_gt_data([RGBout_img, gt_long_img])
                
                # ========= pack output ================
                if self.Training_config.phase == 'deblur':
                    outputs = pack_network_output(outs, self.opt.DeblurNet.name)
                elif self.Training_config.phase == 'denoise':
                    outputs = pack_network_output(outs, self.opt.DenoiseNet.name)

                # =========== train G ================
                G_loss, G_loss_info = self.LM(outputs, gt_dict, None)

                self.optim_G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

                # record loss
                if iters_done % self.Training_config.show_loss_iter == 0:
                    self.add_scalars(main_tag = 'G_loss', tag_scalar_dict = G_loss_info, global_step = iters_done)
                if iters_done % self.Training_config.show_img_iter == 0:
                    if self.Training_config.phase == 'deblur':
                        vis_imgs = [down_short_img, down_long_img, down_out_img]
                    elif self.Training_config.phase == 'denoise':
                        vis_imgs = [short_img, long_img, RGBout_img, deblur_out]
                    
                    if isinstance(outs, list):
                        for j in range(len(outs)):
                            if outs[j] is not None and outs[j].size()[1] == 3:
                                vis_imgs.append(outs[j].clamp(0.0, 1.0))
                    else:
                        outs = outs.clamp(0.0, 1.0)
                        vis_imgs.append(outs)
                    
                    self.visual_image('train_img', vis_imgs, iters_done)

                self._save_G(self.opt, epoch, iters_done, len(self.train_loader), self.G)
                if self.LM.train_GAN:
                    self._save_D(self.opt, epoch, iters_done, len(self.train_loader), self.D)

                self._adjust_learning_rate(self.optim_config, (epoch + 1), iters_done, self.optim_G, self.opt.Optimizer.args.lr_g)
                if self.LM.train_GAN:
                    self._adjust_learning_rate(self.optim_config, (epoch + 1), iters_done, self.optim_D, self.opt.Optimizer.args.lr_d)

                iters_done += 1
            
            self._validate(epoch)


    def _validate(self, epoch):
        self.G.eval()
        val_PSNR, val_SSIM, num_of_val_image = 0, 0, 0

        for j, data in enumerate(self.val_loader):

            # ========= Get data ==================
            short_img = data['short_img'].to(self.device)
            long_img = data['long_img'].to(self.device)
            RGBout_img = data['RGBout_img'].to(self.device)
            gt_long_img = data['gt_long_img'].to(self.device)

            if len(short_img.shape) == 5:
                _, _, C, H, W = short_img.shape
                short_img = short_img.view(-1, C, H, W)
                long_img = long_img.view(-1, C, H, W)
                RGBout_img = RGBout_img.view(-1, C, H, W)
                gt_long_img = gt_long_img.view(-1, C, H, W)

            down_short_img, down_long_img, down_out_img, down_gtlong_img = \
                self.train_loader.dataset.downsample_tensors([short_img, long_img, RGBout_img, gt_long_img])

            # =========== forward ==================
            with torch.no_grad():
                if self.Training_config.phase == 'deblur':
                    print(down_short_img.shape, down_long_img.shape)
                    outs = self.G(down_short_img, down_long_img)
                elif self.Training_config.phase == 'denoise':
                    deblur_out = self.deblurNet(down_short_img, down_long_img).detach()
                    deblur_out = F.upsample(deblur_out, size = (H, W), mode = 'bilinear', align_corners = False)
                    deblur_out = deblur_out.clamp(0.0, 1.0)
                    short_img, long_img, RGBout_img, deblur_out = \
                        self.train_loader.dataset.crop_tensor_patch([short_img, long_img, RGBout_img, deblur_out])
                    outs = self.G(short_img, long_img, deblur_out)

            if isinstance(outs, list):
                out = outs[0]
            else:
                out = outs

            out = out.clamp(0.0, 1.0)

            num_of_val_image += out.shape[0]

            if self.Training_config.phase == 'deblur':
                val_PSNR += utils.psnr(out, down_out_img, 1) * out.shape[0]
                val_SSIM += utils.ssim(out, down_out_img) * out.shape[0]
            elif self.Training_config.phase == 'denoise':
                val_PSNR += utils.psnr(out, RGBout_img, 1) * out.shape[0]
                val_SSIM += utils.ssim(out, RGBout_img) * out.shape[0]
            
            if j % 10 == 0:
                print('val: %d | epoch: %d' % (j, epoch))
        
        val_PSNR = val_PSNR / num_of_val_image
        val_SSIM = val_SSIM / num_of_val_image

        self.add_scalar('val_PSNR', val_PSNR, global_step = epoch)
        self.add_scalar('val_SSIM', val_SSIM, global_step = epoch)

        self.G.train()
