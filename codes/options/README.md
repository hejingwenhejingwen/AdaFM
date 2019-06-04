# Configurations
- Use **json** files to configure options.
- Convert the json file to python dict.
- Support `//` comments and use `null` for `None`.

## Table
Click for detailed explanations for each json file.

1. [test.json](#test_json) 
1. [train_basic.json](#train_basic_json)
1. [train_adafm.json](#train_adafm_json) 


## test_json
### normal testing
please see the example config file [test](test/test.json)
```c++
{
  "name": "test_001_adafmnet_noise75_DIV2K"
  , "suffix": null
  , "model": "sr"
  , "crop_size": 0  // 0 for image restoration | upscale (x2,3,4) for SR
  , "gpu_ids": [0]

  , "interpolate_stride": null // 0.1, 0.05, 0.01, ... for modulation testing

  , "datasets": {
    "test_1": { // the 1st test dataset
      "name": "CBSD68"
      , "mode": "LRHR"
      , "dataroot_HR": "../datasets/val_CBSD68/CBSD68" // path for HR images
      , "dataroot_LR": "../datasets/val_CBSD68/CBSD68_noise75" // path for LR images
    }
    , "test_2": { // the 2nd test dataset
      "name": "personal_images"
      , "mode": "LR"
      , "dataroot_LR": "../datasets/personal_images/personal_images_noise75"
    }
  }

  , "path": {
    "root": "../"
    , "pretrain_model_G": "../experiments/debug_001_adafmnet_noise75_DIV2K/models/8_G.pth" // path for the trained model
  }

  , "network_G": {
    "which_model_G": "adaptive_resnet"
    , "norm_type": "adafm" // basic | adafm | null | instance | batch
    , "nf": 64
    , "nb": 16
    , "in_nc": 3
    , "out_nc": 3
    , "adafm_ksize": 1 // 1 | 3 | 5 | 7
  }
}
```
### modulation testing
for modulation testing, you should specify the interpolation stride:
```c++
, "interpolate_stride": 0.1 // 0.1 | 0.05 | 0.01
```

## train_basic_json
To train a basic model, please modify the [train basic](train/train_basic.json).
```c++
{
  "name": "debug_001_basicmodel_noise15_DIV2K"  // !!! please remove "debug_" during training
  , "use_tb_logger": true  // use tensorboard
  , "model":"sr"
  , "finetune_norm": false  // whether finetune the adafm layers
  , "crop_size": 0 // 0 for image restoration | upscale (x2, x3, x4) for SR
  , "gpu_ids": [0] // gpu id list

  , "datasets": {
    "train": {
      "name": "DIV2K"
      , "mode": "LRHR"
      , "dataroot_HR": "../datasets/DIV2K800/DIV2K800_sub" // path for HR images
      , "dataroot_LR": "../datasets/DIV2K800/DIV2K800_sub_noise15" // path for LR images
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 16 // batch size
      , "HR_size": 96 // crop szie for the HR image
      , "use_flip": true
      , "use_rot": true
    }
    , "val": {
      "name": "val_CBSD68"
      , "mode": "LRHR"
      , "dataroot_HR": "../datasets/val_CBSD68/CBSD68" // path for HR images
      , "dataroot_LR": "../datasets/val_CBSD68/CBSD68_noise15" // path for LR images
    }
  }

  , "path": {
    "root": "../" // the path root for the current experiment
    // , "resume_state": "../experiments/debug_001_basicmodel_noise15_DIV2K/training_state/200.state"
    , "pretrain_model_G": null // path for pretrained model
  }

  , "network_G": {
    "which_model_G": "adaptive_resnet"
    , "norm_type": "basic" // basic | adafm | null | instance | batch
    , "nf": 64 // the number of the channel
    , "nb": 16 // the number of the residual blocks
    , "in_nc": 3 // the number of the input channel
    , "out_nc": 3 // the number of the output channel
    , "adafm_ksize": null // the filter size of adafm during finetune
  }

  , "train": {
    "lr_G": 1e-4 // learning rate
    , "lr_scheme": "MultiStepLR" // learning rate decay scheme
    , "lr_steps": [500000] // at which steps, decay the learining rate
    , "lr_gamma": 0.1 // learning rate decreases by a factor of 0.1

    , "pixel_criterion": "l1" // l1 loss
    , "pixel_weight": 1.0 // the weight of l1 loss
    , "val_freq": 5e3 // how often do you want to do validation

    , "manual_seed": 0
    , "niter": 1e6 // the total number of the training iterations
  }

  , "logger": {
    "print_freq": 200 // how often to log the training stats
    , "save_checkpoint_freq": 5e3 // how often to save the checkpoints
  }
}
```

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

