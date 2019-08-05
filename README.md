# Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers [paper](https://arxiv.org/abs/1904.08118), [supplementary file](http://openaccess.thecvf.com/content_CVPR_2019/supplemental/He_Modulating_Image_Restoration_CVPR_2019_supplemental.pdf)
By Jingwen He, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), and [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/)

<p align="center">
  <img height="250" src="./figures/framework.PNG">
</p>

```python
class AdaptiveFM(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(AdaptiveFM, self).__init__()
        padding = (kernel_size - 1) // 2
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding, groups=in_channel)

    def forward(self, x):
        return self.transformer(x) + x
```

### BibTex
    @InProceedings{He_2019_CVPR,
    author = {He, Jingwen and Dong, Chao and Qiao, Yu},
    title = {Modulating Image Restoration With Continual Levels via Adaptive Feature Modification Layers},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    }

### Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 0.4.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.

# Pretrained models
We provide a pretrained model for AdaFM-Net ([`experiments/pretrained_models`](experiments/pretrained_models)) that deals with denoising from σ15 to σ75. Please run the following commands directly:
```c++
cd codes
python interpolate.py -opt options/test/test.json
```
The results can be found in the newly created directory `AdaFM/results` 
The noise level of the [`input image`](datasets/personal_images/personal_images_noise45/soilder.png) is σ45, and you are supposed to obtain similar interpolated results as follows:

<p align="center">
  <img height="100" src="./figures/modulation_testing.PNG">
</p>

# Codes
The overall code framework mainly consists of four parts - Config, Data, Model and Network.
We also provides some useful scripts. 
Please run all the following commands in “codes” directory.

## How to Test

### basic model and AdaFM-Net
1. Modify the configuration file [`options/test/test.json`](codes/options/test/test.json) (please refer to [`options`](codes/options) for instructions.)
1. Run command:
```c++
python test.py -opt options/test/test.json
```

### modulation testing
1. Modify the configuration file [`options/test/test.json`](codes/options/test/test.json) 
1. Run command:
```c++
python interpolate.py -opt options/test/test.json
```
#### or:
1. Use [`scripts/net_interp.py`](codes/scripts/net_interp.py) to obtain the interpolated network.
1. Modify the configuration file [`options/test/test.json`](codes/options/test/test.json) and run command: `python test.py -opt options/test/test.json`

## How to Train

### basic model
1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](codes/data). 
1. Modify the configuration file [`options/train/train_basic.json`](codes/options/train/train_basic.json) (please refer to [`options`](codes/options) for instructions.)
1. Run command: 
```c++
python train.py -opt options/train/train_basic.json
```
### AdaFM-Net
1. Prepare datasets, usually the DIV2K dataset.
1. Modify the configuration file [`options/train/train_adafm.json`](codes/options/train/train_adafm.json)
1. Run command:
```c++
python train.py -opt options/train/train_adafm.json
```

## Acknowledgement

- This code borrows heavily from [BasicSR](https://github.com/xinntao/BasicSR).
