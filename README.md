# Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers [paper](https://arxiv.org/abs/1904.08118)

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
    @InProceedings{he2019modulating,
    author = {He, Jingwen and Dong, Chao and Qiao, Yu},
    title = {Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers},
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

### Pretrained models
We provide a model of AdaFM-Net ([`experiments/pretrained_models`](experiments/pretrained_models)) that deals with denoising from σ15 to σ75. Please run the following command directly:
```c++
python interpolate.py -opt options/test/test.json
```
The noise level of the input image is σ45, you are supposed to obtain similar interpolated results as follows:

<p align="center">
  <img height="100" src="./figures/modulation_testing.PNG">
</p>

# Codes
The overall code framework mainly consists of four parts - Config, Data, Model and Network.
We also provides some useful scripts. 

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

- The code is based on [BasicSR](https://github.com/xinntao/BasicSR) framework.
