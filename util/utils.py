import os
import torch
import numpy as np
import cv2
import skimage
import skimage.measure


# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, save_format = 'png'):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.' + save_format
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)


def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p


def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p


def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim


# ----------------------------------------
#             PATH processing
# ----------------------------------------
def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)


def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret


def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret


def get_jpgs_once(paths):
    # read a folder, return the image name
    ret = set([])
    if isinstance(paths, str):
        paths = [paths]
    assert isinstance(paths, list)
    for path in paths:
        for root, dirs, files in os.walk(path):
            for filespath in files:
                filespath = filespath.split('_')[0] + '_' + filespath.split('_')[1]
                filespath = os.path.join(root, filespath)
                if filespath not in ret:
                    ret.add(filespath)
    ret = list(ret)
    return ret


def get_blur_file_once(paths):
    # read a folder, return the image name
    ret = set([])
    if isinstance(paths, str):
        paths = [paths]
    assert isinstance(paths, list)
    for path in paths:
        for root, dirs, files in os.walk(path):
            for filespath in files:
                filespath = filespath.split('_')[0] + '_' + filespath.split('_')[1] + '_' + filespath.split('_')[2]
                filespath = os.path.join(root, filespath)
                if filespath not in ret:
                    ret.add(filespath)
    ret = list(ret)
    return ret


def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content


def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_file(path):
    return os.path.exists(path)


def normalize_ImageNet_stats(batch):
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch_out = (batch - mean) / std
    return batch_out
