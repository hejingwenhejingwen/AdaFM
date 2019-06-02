# Configurations
- Use **json** files to configure options.
- Convert the json file to python dict.
- Support `//` comments and use `null` for `None`.

## Table
Click for detailed explanations for each json file.

1. [train_basic.json](#train_basic_json)
1. [train_adafm.json](#train_adafm_json) 
1. [test.json](#test_json) 


## train_basic_json
To train a basic model, please modify the [train basic](train/train_basic.json).

## train_adafm_json
To finetune the AdaFM layers in AdaFM-Net, you need at least modify the followings in [train adafm](train/train_adafm.json).

- "name": "THE_NAME".
- "finetune_norm": true.
- the "dataroot_LR" for both "train" and "val". 
```c++
 , "datasets": {
    "train": {
      "name": "DIV2K"
      , "mode": "LRHR"
      , "dataroot_HR": "../datasets/DIV2K800/DIV2K800_sub" 
      , "dataroot_LR": "../datasets/DIV2K800/DIV2K800_sub_noise75" // path for LR images
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 16
      , "HR_size": 96
      , "use_flip": true
      , "use_rot": true
    }
   , "val": {
      "name": "val_CBSD68"
      , "mode": "LRHR"
      , "dataroot_HR": "../datasets/val_CBSD68/CBSD68" 
      , "dataroot_LR": "../datasets/val_CBSD68/CBSD68_noise75" // path for LR images
    }
```
- "pretrain_model_G" for path of basic model:
```c++
"path": {
   "root": "../" 
   , "pretrain_model_G": "../experiments/debug_001_basicmodel_noise15_DIV2K/models/1000000_G.pth" // the path for basic model
 }
```
- the "norm_type" and the "adafm_ksize" in "network_G":
```c++
, "network_G": {
    "which_model_G": "adaptive_resnet"
    , "norm_type": "adafm" // basic | adafm | null | instance | batch
    , "nf": 64
    , "nb": 16
    , "in_nc": 3
    , "out_nc": 3
    , "adafm_ksize": 1 // the filter size of adafm during finetune. 1 | 3 | 5 | 7
  }
```

#test_json
###normal testing
please see the example config file [test](test/test.json)

###modulation testing
for modulation testing, you should specify the interpolation stride:
```c++
, "interpolate_stride": 0.1 // 0.1 | 0.05 | 0.01
```