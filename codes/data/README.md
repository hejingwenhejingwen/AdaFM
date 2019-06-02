
Dataloader

- use opencv (`cv2`) to read and process images.

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`codes/scripts/create_lmdb.py`](../scripts/create_lmdb.py).

## Contents

- `LR_dataset`: only reads LR images in test phase where there is no GT images.
- `LRHR_dataset`: reads LR and HR pairs from image folder or lmdb files. Used in SR, DeJPEG, Denoising training and validation phase.


## How To Prepare Data
### SR, DeJPEG, Denoising
1. Prepare the images. You can download DIV2K dataset can be downloaded from [DIV2K offical page](https://data.vision.ee.ethz.ch/cvl/DIV2K/), or from [Baidu Drive](https://pan.baidu.com/s/1LUj90_skqlVw4rjRVeEoiw).

1. We use DIV2K dataset for training the SR, DeJPEG, and Denoising models. 
    1. since DIV2K images are large, we first crop them to sub images using [`codes/scripts/extract_subimgs_single.py`](../scripts/extract_subimgs_single.py). 
    1. generate LR images using matlab with [`codes/scripts/generate_mod_LR_bic.m`](../scripts/generate_mod_LR_bic.m), [`codes/scripts/generate_LR_JPEG.m`](../scripts/generate_LR_JPEG.m), and [`codes/scripts/generate_LR_noise.m`](../scripts/generate_LR_noise.m). If you already have LR images, you can skip this step. Please make sure the LR and HR folders have the same number of images.
    1. generate .lmdb file if needed using [`codes/scripts/create_lmdb.py`](../scripts/create_lmdb.py).
    1. modify configurations in `options/train/xxx.json` when training, e.g., `dataroot_HR`, `dataroot_LR`.



4. The same for validation (you can choose some from the test folder) and test folder.

## General Data Process

### data augmentation

We use random crop, random flip/rotation, (random scale) for data augmentation. 
