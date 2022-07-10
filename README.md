# D2HNet

## 1 Introduction

This project is a night image restoration framework called D2HNet by jointly denoising and deblurring successively captured long- and short-exposure images. To train and benchmark D2HNet, we create a large-scale D2-Dataset.

![img1](./img/img1.png)

## 2 Dependency

This code is based on PyTorch 1.1 and CUDA 9.0. It has been tested on Ubuntu 18.04 LTS, where the machine is equipped with NVIDIA Titan GPUs.

We use Anaconda to set up the environment. Users can set up a new environment simply by:

```bash
conda env create -f environment.yml
conda activate d2hnet
```

Then, users need to install the deformable convolution module (thanks to previous [implementation1](https://github.com/xinntao/EDVR/tree/master/basicsr/models/ops/dcn) and [implementation2](https://github.com/JimmyChame/LSFNet/tree/master/dcn)) in the `d2hnet` environment by:

```bash
cd dcn
python setup.py develop
```

If you got an error when installing deformable convolution module, please delete the `build` folder first and then run:

```bash
cd dcn
rm -rf build
python setup.py build develop
```

## 3 Dataset

### 3.1 D2-Dataset

Users can find the full D2-Dataset through the [link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EgAgxXmhdExEm53S5vlkfeABg5ACBcYxvcyr7gMDWzS0sw?e=EK1sKa). Just unzip each `zip` file in each path. Since the full dataset is quite large (554Gb), users may download seperated dataset, and the links are given below.

| Name | Description | Resolution | Numbers | Link and Size |
| ---- | ---- | :----: | :----: | :----: |
| original/train | synthetic training tuples | 2560x1440 | 5661 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EWJ09tx5QUdPrmvKM3Ne3m4B5J47MQVXHxutv_3_ARUh4A?e=ZKNWYG) 192 GB |
| original/train_blur_patch | selected patch tuples by VarmapSelection from `train` | 1024x1024 | 9453 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EYqegoikxoxJoxy-qKqTrU4BWwcBrwOcc0DLclfqSJzymw?e=qhHHeg) 99.0 GB |
| original/val | synthetic validation tuples from videos 1-13 | 2560x1440 | 268 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EQum2_I9mL5MltdFXciztkwBhxf1o2qPDL3iNmkL20KdyQ?e=SLrdfj) 9.44 GB |
| original/val_no_overlap | synthetic validation tuples from videos 14-30 | 2560x1440 | 1192 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/Edx1bBZ1znVNsDSbrLUub10BxGU72mfqO0tvNOIvS5XMRg?e=dxwm4X) 38.7 GB |
| original/val_no_overlap_blur_patch | selected patch tuples by VarmapSelection from `val_no_overlap` | 1024x1024 | 99 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EbC3GB8PFitIvgo0SJbpj90BhFwimWmXqfcsmMl9j8gG0w?e=uwdMsZ) 1.11 GB |
| original/val_no_overlap_noisy_1440p | noisy tuples by adding calibrated noises to `val_no_overlap` | 1024x1024 | 1192 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/ETo2YmEvSDhKk0ETBec6ZF8BvseovbL-HbKiqZu3W391TA?e=oUejhu) 17.6 GB |
| sharpened/train | MATLAB sharpened `train` tuples | 2560x1440 | 5661 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/Eeg_AdGiTAxGmmbaLBvwPcoBiExyxLylBqoUfppV18iFVw?e=nSeghK) 124 GB |
| sharpened/train_blur_patch | MATLAB sharpened `train_blur_patch` tuples | 1024x1024 | 9453 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EfaX81jkGFtLq1B1_p3eR2MBu37f92ziZuQCwOJCw8B5sw?e=zp8h2Y) 59.0 GB |
| sharpened/val | MATLAB sharpened `val` tuples | 2560x1440 | 268 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EbAN8RMlvNhMmniGIVV-XmcBeMHojt3YaOFkn46Y3DRaXw?e=gQ0qa6) 6.08 GB |
| sharpened/val_no_overlap | MATLAB sharpened `val_no_overlap` tuples | 2560x1440 | 1192 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EUdc3tybG2dGiGkX8I8rxnoBZq0uGuA8o5KB95bBFoeDNw?e=CNvKKS) 23.2 GB |
| sharpened/val_no_overlap_blur_patch  | MATLAB sharpened `val_no_overlap_blur_patch` tuples | 1024x1024 | 99 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EV-dd99FPalEolcfoVasSCYBsPX0TeTuxMAiZBM0QIKvQA?e=QwEjB9) 659 MB |
| Xiaomi_Mi_Note_10_photos | real captured long- and short-exposure photos | 4624x3472 | 28 | [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EYshIGNbXwlBhAm_gx-uoVoBKqTxBBvXYkdvaCSpkWpCrw?e=GLDUhk) 2.45 GB |

Please unzip it in this path. Then, rename the full path to `data`.

### 3.2 Image capturing tool (Andriod apk)

To appear soon

## 4 Train

### 4.1 Run

D2HNet has two subnets (DeblurNet and EnhanceNet) which are trained sequentially. But they share the same entering file:

```bash
python train.py
```

Users need to change the parameters of `train.py` to train a specific network, as follows:

| Name | Description | Training DeblurNet | Training EnhanceNet |
| ---- | ---- | :----: | :----: |
| --opt | network to be trained | 'options/tp_deblurnet_v2_002.yaml' | 'options/tp_denoisenet_v2_002.yaml' |
| --num_gpus | numbers of GPUs to be used | 2 | 2 |
| --save_path | saving path to trained models | 'snapshot/tp_deblurnet_v2_002' | 'snapshot/tp_denoisenet_v2_002' |
| --log_path | saving path to tensorboard results | 'log_pt/tp_deblurnet_v2_002' | 'log_pt/tp_denoisenet_v2_002' |

### 4.2 Hyper-parameters of the network

Users may change some parameters in `yaml` files to fit their machine and requirement:

- vgg_model_path: if users want to add the perceptual loss, please download VGG-16 officially pre-trained model: https://download.pytorch.org/models/vgg16-397923af.pth. Then, put it under `pretrained_models` folder. Otherwise, please comment out the `VGGLoss` in the `yaml` files.
- train_path: path to original synthetic training set (`original/train`)
- val_path: path to original synthetic validation set (`original/val_no_overlap`)
- train_sharp_path: path to sharpened training set (`sharpened/train`)
- val_sharp_path: path to sharpened validation set (`sharpened/val_no_overlap`)
- deblur_crop_size: size of the cropped square from a full-resolution image for DeblurNet
- deblur_patch_per_image: the numbers of cropped patches from a full-resolution image for DeblurNet
- deblur_size: the processing resolution of DeblurNet at training
- denoise_crop_size: the processing resolution of EnhanceNet at training, cropped from DeblurNet results
- denoise_patch_per_image: the numbers of cropped patches for EnhanceNet
- cutblur: CutNoise-related hyper-parameters
- darken: Illumination Adjustment (IA)-related hyper-parameters
- color_distor: Color Adjustment (CA)-related hyper-parameters

### 4.3 D2HNet pre-trained models

Users can download pre-trained models via this [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/ESpmAMBVXlpOl1bgD3i1DAQB2AmleZ-nHxxNjh2GmrguTg?e=mg6Ff6).

Please unzip it in this path, you will get a folder named `snapshot`.

## 5 Val and Test

### 5.1 Validation

Noisy long- and short-exposure validation pairs with different resolutions (e.g., 1440x2560 or 2880x5120) are pre-generated. We provide a sample `original/val_no_overlap_noisy_1440p` for running the script. Please change `--val_path` and `--val_sharp_path` to the paths to input (e.g., `original/val_no_overlap_noisy_1440p`) and ground truth (`sharpened/val_no_overlap`), respectively. Users can test the provided models using the following script:

```bash
python validation.py
```

### 5.2 Testing (on Real-world Photos)

Noisy long- and short-exposure real-world pairs are provided. Please change `--src_path` to the data path (`Xiaomi_Mi_Note_10_photos`). Users can test the provided models using the following script:

```bash
python realworld.py
```

## 6 Citation

If you find this work useful for your research, please cite:

```bash
@article{zhao2022d2hnet,
  title={D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration},
  author={Zhao, Yuzhi and Xu, Yongzhe and Yan, Qiong and Yang, Dingdong and Wang, Xuehui and Po, Lai-Man},
  journal={arXiv preprint arXiv:2207.03294},
  year={2022}
}

@inproceedings{zhao2022d2hnet,
  title={D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration},
  author={Zhao, Yuzhi and Xu, Yongzhe and Yan, Qiong and Yang, Dingdong and Wang, Xuehui and Po, Lai-Man},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
