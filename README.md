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

    @misc{he2019modulating,
    title={Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers},
    author={Jingwen He and Chao Dong and Yu Qiao},
    year={2019},
    eprint={1904.08118},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }

### Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 0.4.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.

# Codes
The overall code framework mainly consists of four parts - Config, Data, Model and Network.
We also provides some useful scripts. 

## How to Test

### basic model and AdaFM-Net
1. Modify the configuration file `options/test/test.json` (refer to [`options`](codes/options))
1. Run command: `python test.py -opt options/test/test.json`

### modulation testing
1. Modify the configuration file `options/test/test.json` 
1. Run command: `python interpolate.py -opt options/test/test.json`
#### or:
1. Use [`scripts/net_interp.py`](codes/scripts/net_interp.py) to obtain the interpolated network.
1. Modify the configuration file `options/test/test.json` and run command: `python test.py -opt options/test/test.json`

## How to Train

### basic model
1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](codes/data). 
1. Modify the configuration file `options/train/train_basic.json`
1. Run command: `python train.py -opt options/train/train_basic.json`

### AdaFM-Net
1. Prepare datasets, usually the DIV2K dataset.
1. Modify the configuration file `options/train/train_adafm.json`
1. Run command: `python train.py -opt options/train/train_adafm.json`


## Acknowledgement

- The code is based on [BasicSR](https://github.com/xinntao/BasicSR) framework.
