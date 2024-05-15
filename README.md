# [PBVS @ CVPR 2024 Thermal Images Super-resolution challenge- Track 1]

Our method achieved 4th place in PSNR and 3rd place in SSIM in the [PBVS 2024 Thermal Image Super-Resolution Challenge(TISR)-Track1](https://codalab.lisn.upsaclay.fr/competitions/17013#results).

## Environment
- Python == 3.8.0
- [PyTorch == 1.11.0](https://pytorch.kr/get-started/previous-versions/)
- BasicSR == 1.3.4.9
- eninops

## Installation
- Install Pytorch, BasicSR and Einops first. Then,
```
python setup.py develop
```
## Data Preparation
### DF2K for Pre-training
- Preparation of DF2K dataset can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). save it in `./datasets`

Extract sub-images
```
python df2k_extract_subimages.py
```
Create a meta-info file
```
python df2k_generate_meta_info.py
```
### Thermal Image for Fine-tuning
- Preparation of Thermal Image can refer to [this page](https://codalab.lisn.upsaclay.fr/competitions/17013#learn_the_details).

Extract sub-images
```
python thermal_extract_subimages.py
```
Create a meta-info file
```
python thermal_generate_meta_info.py
```
## Quick[test]
- Refer to `./options/test`
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing). Save it in ./experiments/pretrained/. (best performance at 14,000 iterations)

Create SR images
```
python hat/test.py -opt options/test/HAT_SRx8_quick.yml
```
The testing results will be saved in the `./results` folder.

## Pre-training
- Refer to `./options/train`
Pretraining command
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_pre.yml --launcher pytorch
```
The training logs and weights will be saved in the `./experiments` folder.

## Fine-tuning
- Refer to `./options/train`
- The model pre-trained on the DF2K dataset is available at
[Google Drive](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing). Save it in ./experiments/pretrained/. (We conducted training for 191,000 iterations.)
Fine-tuning command
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_48_cutblur_fineturn.yml --launcher pytorch
```
The training logs and weights will be saved in the `./experiments` folder.

## Acknowledgment
Our codes borrowed from [chxy95](https://github.com/XPixelGroup/HAT) and [nmhkahn](https://github.com/clovaai/cutblur). Thanks for their work.
