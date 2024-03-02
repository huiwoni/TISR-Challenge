## Environment
- PyTorch == 1.11.0
- BasicSR == 1.3.4.9
- einops

### Installation
Install Pytorch, BasicSR, einops.

```
python setup.py develop
```
## Quick[test]
- Refer to `./options/test`
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing). Save it in ./experiments/pretrained/.

- Create SR images
```
python hat/test.py -opt options/test/HAT_SRx8_quick.yml
```
The testing results will be saved in the `./results` folder.

# Pre-training
- Refer to `./options/train`
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).
- The pretraining command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_pre.yml --launcher pytorch
```
-The training logs and weights will be saved in the `./experiments` folder.

# Fine-tuning
- Refer to `./options/train`
- Preparation of training data can refer to [this page](https://codalab.lisn.upsaclay.fr/competitions/17013#learn_the_details).
- The model pre-trained on the DF2K dataset is available at
[Google Drive](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing). Save it in ./experiments/pretrained/.
- The Fine-tuning command is like
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_48_cutblur_fineturn.yml --launcher pytorch
```
-The training logs and weights will be saved in the `./experiments` folder.

# Acknowledgment
Our codes borrowed from [XPixelGroup](https://github.com/XPixelGroup/HAT). 
