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

Users can find D2-Dataset through the [link](). Just unzip each `zip` file in each path. Since the full dataset is quite large (554Gb), users may download seperated dataset, and the links are given below.

| Name | Description | Resolution | Numbers | Path(.zip) |
| ---- | ---- | :----: | :----: | ---- |
| original/train | synthetic training tuples | 2560x1440 | 5661 | [link]() |
| original/train_blur_patch | selected patch tuples by VarmapSelection | 1024x1024 | 9453 | [link]() |
| original/val | synthetic validation tuples from videos 1-13 | 2560x1440 | 268 | [link]() |
| original/val_no_overlap | synthetic validation tuples from videos 14-30 | 2560x1440 | 1192 | [link]() |
| sharpened/train | MATLAB sharpened `train` tuples | 2560x1440 | 5661 | [link]() |
| sharpened/train_blur_patch | MATLAB sharpened `train_blur_patch` tuples | 1024x1024 | 9453 | [link]() |
| sharpened/val | MATLAB sharpened `val` tuples | 2560x1440 | 268 | [link]() |
| sharpened/val_no_overlap | MATLAB sharpened `val_no_overlap` tuples | 2560x1440 | 1192 | [link]() |
| Xiaomi_Mi_Note_10_photos | real captured long- and short-exposure photos | 4624x3472 | 28 | [link]() |

### 3.2 Image capturing tool (Andriod apk)

To appear soon

## 4 Train

### 4.1 Run

D2HNet has two subnets which are trained sequentially:

```bash
python train.py
```

Setting parameters:

| Name | Description | Training DeblurNet | Training EnhanceNet |
| ---- | ---- | :----: | :----: |
| --opt | network to be trained | 'options/tp_deblurnet_v2_002.yaml' | 'options/tp_denoisenet_v2_002.yaml' |
| --num_gpus | numbers of GPUs to be used | 2 | 2 |
| --save_path | saving path to trained models | 'snapshot/tp_deblurnet_v2_002' | 'snapshot/tp_denoisenet_v2_002' |
| --log_path | saving path to tensorboard results | 'log_pt/tp_deblurnet_v2_002' | 'log_pt/tp_denoisenet_v2_002' |

### 4.2 Hyper-parameters of the network

Users may change some parameters in `yaml` files to fit their machine:

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

To appear soon

## 5 Val and Test

To appear soon

## 6 Citation

If you find this work useful for your research, please cite:

```bash
@inproceedings{zhao2022d2hnet,
  title={D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration},
  author={Zhao, Yuzhi and Xu, Yongzhe and Yan, Qiong and Yang, Dingdong and Wang, Xuehui and Po, Lai-Man},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```