import numpy as np
from PIL import Image, ImageEnhance
import cv2
import random
import colour_demosaicing
import math
from scipy.stats import tukeylambda

from .noise import VIVO_NOISE, MI_NOTE10_NOISE, MI_NOTE10_NOISE_20220330


def adjust_gamma(image, gamma = 2.2):
    """
    image should be in [0, 1]
    """
    return np.power(image, gamma)


def cutblur(d_image, gt_image, cut_size, cut_prob):
    """
    args:
        d_image: NHWC or HWC
        gt_image: NHWC or HWC
    """
    if len(d_image.shape) == 3:
        p = random.random()
        if p < cut_prob:
            h, w = d_image.shape[:2]
            rand_h = random.randint(0, h - cut_size)
            rand_w = random.randint(0, w - cut_size)
            d_image[rand_h:rand_h+cut_size, rand_w:rand_w+cut_size+cut_size] = \
                gt_image[rand_h:rand_h+cut_size, rand_w:rand_w+cut_size+cut_size]
    elif len(d_image.shape) == 4:
        for i in range(d_image.shape[0]):
            p = random.random()
            if p < cut_prob:
                h, w = d_image[i].shape[:2]
                rand_h = random.randint(0, h - cut_size)
                rand_w = random.randint(0, w - cut_size)
                d_image[i, rand_h:rand_h+cut_size, rand_w:rand_w+cut_size+cut_size] = \
                    gt_image[i, rand_h:rand_h+cut_size, rand_w:rand_w+cut_size+cut_size]
    else:
        raise ValueError('shape [%s] of d_image is not supported' % str(d_image.shape))
    return d_image


def curblur_tensor(d_image, gt_image, cut_size, cut_prob):
    """
    cutblur function for pytorch tensor
    args: 
        d_image: NCHW
        gt_image: NCHW
    """
    h, w = d_image.shape[2:]
    for i in range(d_image.shape[0]):
        p = random.random()
        if p < cut_prob:
            rand_h = random.randint(0, h - cut_size)
            rand_w = random.randint(0, w - cut_size)
            d_image[i, :, rand_h:rand_h+cut_size, rand_w:rand_w+cut_size] = \
                gt_image[i, :, rand_h:rand_h+cut_size, rand_w:rand_w+cut_size]

    return d_image


def cutout_tensor(d_image, cut_size, cut_prob):
    """
    cutout function for pytorch tensor
    args:
        d_image: NCHW
    """
    h, w = d_image.shape[2:]
    for i in range(d_image.shape[0]):
        p = random.random()
        if p < cut_prob:
            rand_h = random.randint(0, h - cut_size)
            rand_w = random.randint(0, w - cut_size)
            d_image[i, :, rand_h:rand_h+cut_size, rand_w:rand_w+cut_size] = 0.
    
    return d_image


def apply_darken(img_array, enhance_factor, gamma_factor):
    """
    img_array: HWC, pixel should be in [0,1]
    """
    img_array = (img_array * 255.).astype(np.uint8)
    pil_img = Image.fromarray(img_array) 
    brightness_enhance = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhance.enhance(enhance_factor)
    
    img_array = np.asarray(pil_img)
    img_float = img_array.astype(np.float32) / 255.0
    img_float = img_float ** gamma_factor
    # img_array = (img_float * 255.0).astype(np.uint8)

    return img_float


def darken(images, dark_prob = 0.5):
    """
    images: numpy array, pixel should be in [0, 1]
    """
    p = random.random()
    if p < dark_prob:
        enhance_factor = random.choice([0.6, 0.7, 0.8, 0.9])
        gamma_factor = 1 / random.choice([0.6, 0.7, 0.75, 0.8, 0.9])
        if len(images.shape) == 4:
            temp = []
            for i in range(images.shape[0]):
                image = apply_darken(images[i], enhance_factor, gamma_factor)
                temp.append(image)
            images = np.array(temp)
        elif len(images.shape) == 3:
            images = apply_darken(images, enhance_factor, gamma_factor)
        else:
            raise ValueError('Image shape %s is not supported.' % str(images.shape))

        return images
    else:
        return images

'''
def add_shot_noise(img, iso, ratio=None, noise='mi_note10'):
    """
    img: should be in [0, 1], rgb format, HWC
    iso: list containing two number, e.g. [2000, 6000]
    ratio: float, short_iso / long_iso, e.g. 3. 
    """

    img = adjust_gamma(img, 2.2)
    raw = np.zeros(img.shape[:2], dtype=np.float32)
    raw[::2, ::2] = img[::2, ::2, 0]          # R
    raw[1::2, ::2] = img[1::2, ::2, 1]        # G
    raw[::2, 1::2] = img[::2, 1::2, 1]        # G
    raw[1::2, 1::2] = img[1::2, 1::2, 2]      # B

    # Inverse AWB
    awb_r = random.uniform(1.5, 2.4)
    awb_b = random.uniform(1.5, 2.4)
    raw[::2, ::2] /= awb_r  # awb_r
    raw[1::2, 1::2] /= awb_b   # awb_b

    # Assume now already subtracted black level
    raw = np.clip((raw * 1023), 0, 1023)

    # Adding noise
    r = raw[::2, ::2]
    g = raw[1::2, ::2]  # two g is identical till this step
    b = raw[1::2, 1::2]

    if noise == 'vivo':
        return img
    
    elif noise == 'mi_note10':
        # if mode == 'short':
        #     iso = random.randint(2500, 12000)
        # elif mode == 'long':
        #     iso = random.randint(100, 2500)
        iso = random.randint(iso[0], iso[1])

        noise_factor_r = MI_NOTE10_NOISE[0]
        noise_factor_b = MI_NOTE10_NOISE[1]
        noise_factor_g = MI_NOTE10_NOISE[2]
        
        gamma_r = (iso / 100) * noise_factor_r[0]
        beta_r = (iso / 100) ** 2 * noise_factor_r[1] + noise_factor_r[2]

        gamma_b = (iso / 100) * noise_factor_b[0]
        beta_b = (iso / 100) ** 2 * noise_factor_b[1] + noise_factor_b[2]

        gamma_g = (iso / 100) * noise_factor_g[0]
        beta_g = (iso / 100) ** 2 * noise_factor_g[1] + noise_factor_g[2]

        noise_r = np.random.normal(0, scale=np.sqrt(gamma_r * r + beta_r))
        noise_g1 = np.random.normal(0, scale=np.sqrt(gamma_g * g + beta_g))
        noise_g2 = np.random.normal(0, scale=np.sqrt(gamma_g * g + beta_g))
        noise_b = np.random.normal(0, scale=np.sqrt(gamma_b * b + beta_b))

    raw[::2, ::2] += noise_r                    # R
    raw[1::2, ::2] += noise_g1                  # G
    raw[::2, 1::2] += noise_g2                  # G
    raw[1::2, 1::2] += noise_b                  # B

    raw = np.clip(raw, 0, 1023)

    raw = raw.astype(np.uint16)
    raw = raw.astype(np.float32)
    # AWB
    raw[::2, ::2] *= awb_r  # awb_r
    raw[1::2, 1::2] *= awb_b   # awb_b

    raw = np.clip(raw, 0, 1023).astype(np.uint16)

    demosaicked_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(raw, 'RGGB')

    demosaicked_rgb = np.clip(demosaicked_rgb, 0, 1023)
    demosaicked_rgb = demosaicked_rgb.astype(np.uint16).astype(np.float32)

    demosaicked_rgb = np.clip(demosaicked_rgb / 1023, 0, 1)

    img = adjust_gamma(demosaicked_rgb, 1 / 2.2)
    
    return img
'''

# dingdong 0915 noise model
def add_shot_noise(x, iso, ratio = None, noise = 'mi_note10'):
    """
    img x: should be in [0, 1], rgb format, HWC
    iso: list containing two number, e.g. [2000, 6000]
    ratio: float, short_iso / long_iso, e.g. 3. 
    """
    
    if noise == 'mi_note10':
        NOISE = MI_NOTE10_NOISE
    elif noise == 'mi_note10_20220330':
        NOISE = MI_NOTE10_NOISE_20220330
    else:
        raise NotImplementedError('%s noise is not implemented' % noise)

    iso = random.randint(iso[0], iso[1])

    # Raw container
    raw = np.zeros(x.shape[:2], dtype = np.float32)
    black_level = 64

    # -------------------- Newest One --------------------
    # Quadratic regression from K-iso
    # K_g = iso ** 2.0 * 5.26068e-09 + iso * -1.30090e-04 + 0.94903
    # K_r = iso ** 2.0 * 4.18512e-09 + iso * -1.09092e-04 + 0.88554
    # K_b = iso ** 2.0 * 7.41530e-09 + iso * -1.67515e-04 + 1.11545
    iso_log = math.log10(iso)
    K_g = iso_log ** 2.0 * NOISE["K_g"][0] + iso_log * NOISE["K_g"][1] + NOISE["K_g"][2]
    K_r = iso_log ** 2.0 * NOISE["K_r"][0] + iso_log * NOISE["K_r"][1] + NOISE["K_r"][2]
    K_b = iso_log ** 2.0 * NOISE["K_b"][0] + iso_log * NOISE["K_b"][1] + NOISE["K_b"][2]

    sigma = math.exp(math.log(K_b) * (NOISE["sigma"][0]) + NOISE["sigma"][1])
    '''
    print("iso {0}, K_grb [{1:.3f}, {2:.3f}, {3:.3f}], sigma {4:.2f}"\
            .format(iso, K_g, K_r, K_b, sigma))
    '''

    x = np.power(x, 2.2)                    # Inverse gamma

    # Resampling
    raw[::2, ::2] = x[::2, ::2, 0]          # R
    raw[1::2, ::2] = x[1::2, ::2, 1]        # G
    raw[::2, 1::2] = x[::2, 1::2, 1]        # G
    raw[1::2, 1::2] = x[1::2, 1::2, 2]      # B

    # Inverse AWB, randomly choosing AWB parameters
    #   from red [1.9, 2.4], blue [1.5, 1.9]
    awb_r, awb_b = 0.5 * random.random() + 1.9, 0.4 * random.random() + 1.5
    raw[::2, ::2] /= awb_r  # awb_r
    raw[1::2, 1::2] /= awb_b  # awb_b

    # Assume now already subtracted black level
    raw = np.clip((raw * 1023), 0, 1023)

    # Shot noise(Possion noise)
    raw[::2, ::2]   = np.random.poisson(raw[::2, ::2] * K_r) / K_r    # R
    raw[1::2, ::2]  = np.random.poisson(raw[1::2, ::2] * K_g) / K_g   # G
    raw[::2, 1::2]  = np.random.poisson(raw[::2, 1::2] * K_g) / K_g   # G
    raw[1::2, 1::2] = np.random.poisson(raw[1::2, 1::2] * K_b) / K_b  # B
    raw = np.clip(raw, 0, 1023)

    # Read noise(addtion noise) 
    tl_addition = tukeylambda.rvs(NOISE["lambda"], size = raw.shape, loc = 0.0, scale = sigma) / 0.9375
    raw = raw + tl_addition
    raw = np.clip(raw, 0, 1023)

    # AWB
    raw[::2, ::2] *= awb_r  # awb_r
    raw[1::2, 1::2] *= awb_b   # awb_b
    raw = np.clip(raw, 0, 1023).astype(np.uint16)

    demosaicked_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(raw, 'RGGB')
    demosaicked_rgb = np.clip(demosaicked_rgb / 1023, 0, 1)
    x = np.power(demosaicked_rgb, 1 / 2.2)

    return x


def color_distortion(img, prob = 0.5):
    """
    img: HWC(RGB format) or HW, value in [0, 1]
    """
    p = random.uniform(0.0, 1.0)
    if p < prob:
        img = img.astype(np.float32)
        if len(img.shape) == 2:
            a = random.uniform(0.3, 0.6)
            b = random.uniform(0.001, 0.01)
            img = a * img + b
        elif len(img.shape) == 3:
            a = random.uniform(0.6, 0.9)
            b = random.uniform(0.001, 0.01)
            _, _, C = img.shape
            for i in range(C):
                img[:, :, i] = a * img[:, :, i] + b
        else:
            raise ValueError("img shape %s is not supported." % (img.shape))
        
        return np.clip(img, 0.0, 1.0)
    else:
        return img
