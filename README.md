# ASiM

ASiM is an inference simulation framework for SRAM-based ACiM design developed by Keio CSG. We hope it would help space exploration of ACiM circuit design and fast inference evaluation on a prototype chip. For any discussion, please feel free to comment on the "Issues" tab, thanks.

## 1. Overview

### 1.1. Introduction to the Directory

* `./bins`: Some useful scripts to process datasets and model weights.
* `./dataset`: Datasets for CiM benchmarks.
* `./model`: DNN models for simulations.
* `./module`: ASiM modules for Bit-wise simulation.
* `./tools`: Some useful tools.
* `./main`: Main directory for simulations.

### 1.2. Quick Start

All you need to do is to modify the parameter/argument settings `config.py` in the main folder. Then run `src_train.py` for model training and `src_simulation.py` for simulations.

When you run `src_train.py`, the training log and model weight will be generated in `main/run`.

## 2. Environment

We handle this project with the following environment:

```
python==3.11.7
torch==2.3.0
numpy==1.24.3
matplotlib==3.8.0
easydict==1.13
```

## 3. Simulation Time

A personal GPU will be good to handle most conditions. Simulation can be run with CPU for ResNet-18/CIFAR-10 level task. For ViT simulations on ImageNet, we suggest to run on a GPU server.

ASiM can be run with various devices and an estimation of running time (w/ full validation set) is listed as follows:

| Environment           | Batch Size | Dataset  | Model            | Duration (s) |
|-----------------------|------------|----------|------------------|--------------|
| i9-13980HX (64G DDR5) | 128        | CIFAR-10 | ResNet-18 (W4A4) | 6000         |
| RTX4080-Laptop (12G)  | 1024       | CIFAR-10 | ResNet-18 (W4A4) | 230          |
| RTX4080-Laptop (12G)  | 256        | ImageNet | ResNet-18 (W6A6) | 6645         |
| A6000 (48G)           | 512        | ImageNet | ResNet-18 (W6A6) | 4049         |
| A6000 (48G)           | 1024       | ImageNet | ViT-B-32 (W8A8)  | TODO         |

## 4. ASiM

### 4.1. Abstract

<a href="TODO" target="_blank">Paper Link</a>

TODO

### 4.2. Mode Setting for ASiM Modules
Users can import and replace ASiM modules with default nn.Module into their own DNN models. ASiM modules have additional arguments in module initialization, and all configurable arguments are extracted into `config.py` within this framework. The meaning of each argument can be found in this configuration file. ASiM have two main modes in general use: **Train** and **Simulation**, for model training/fine-tuning and running bit-wise simulation.

#### 4.2.1. Running Simulations by ASiM

For running simulation, a pre-trained model weight has to be specified in `config.py`: `cfg.pretrain_model_path = 'model_path.pkl'`.

The following example shows an ASiM module configuration for simulation:

```
ASiMConv2d(in_planes=64,
           out_planes=128,
           kernel_size=3,
           stride=1,
           padding=1,
           bias=False,
           wbit=8,
           xbit=8,
           adc_prec=8,
           nrow=256,
           rand_noise_sigma=0.05,
           non_linear_sigma=0.05,
           act_enc=None,
           signed_act=False,
           hybrid_levels=None,
           mode='Simulation',
           trim_noise=0.0,
           device='cuda')
```

Please note that the mode must be set as `mode='Simulation'` to switch ASiM modules into simulation mode.

#### 4.2.2. Training DNN Models by ASiM

The following example shows an ASiM module configuration for model training, with mode set as `mode='Train'`:

```
ASiMConv2d(in_planes=64,
           out_planes=128,
           kernel_size=3,
           stride=1,
           padding=1,
           bias=False,
           wbit=8,
           xbit=8,
           adc_prec=8,
           nrow=256,
           rand_noise_sigma=0.0,
           non_linear_sigma=0.0,
           act_enc=None,
           signed_act=False,
           hybrid_levels=None,
           mode='Train',
           trim_noise=0.0,
           device='cuda')
```

The `wbit` and `xbit` have to be specified for quantization-aware training. For training tiny scale DNN models (ResNet-18) with simple tasks (CIFAR-10/CIFAR-100), users can directly train the model quickly. For relately larger models (ViT) and ImageNet task, we recommend to download and process PyTorch official model weight for fine-tuning, with pretrained model weight directory specified as `cfg.pretrain_model_path = 'model_path.pkl'`.

#### 4.2.3. Fine-tuning a Robust DNN Model for ACiM Applications

To perform noise-aware training for obtaining a more robust DNN model, users can configure ASiM modules like:

```
ASiMConv2d(in_planes=64,
           out_planes=128,
           kernel_size=3,
           stride=1,
           padding=1,
           bias=False,
           wbit=8,
           xbit=8,
           adc_prec=8,
           nrow=256,
           rand_noise_sigma=0.0,
           non_linear_sigma=0.0,
           act_enc=None,
           signed_act=False,
           hybrid_levels=None,
           mode='Train',
           trim_noise=50.0,
           device='cuda')
```

Here, a pretrained model weight must be given as `cfg.pretrain_model_path = 'model_path.pkl'`, and training noise intensity must be given like `trim_noise=50.0`. After the noise-aware training, the DNN model can tolerate a certain level of analog noise.

### 4.3. Download Our Post-NAT-Models

| File Name                                | Dataset  | Model            | Link                                                                                                                                                                                |
|------------------------------------------|----------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| resnet18_cifar10_w8a8_pact_trim_100.pkl  | CIFAR-10 | ResNet-18 (W8A8) | <a href="https://www.dropbox.com/scl/fi/qjs2fq6synk38j3h8drd1/resnet18_cifar10_w8a8_pact_trim_100.pkl?rlkey=jemaxi7qhd67rpqgqn2bkfkhs&st=wk17npsf&dl=0" target="_blank">Dropbox</a> |
| resnet18_imagenet_w8a8_pact_trim_50.pkl  | ImageNet | ResNet-18 (W8A8) | <a href="https://www.dropbox.com/scl/fi/4vuwh9ihlclj1vqhk99fg/resnet18_imagenet_w8a8_pact_trim_50.pkl?rlkey=erpc3jbo7kdsdjpu2mhkmtcea&st=gi9ykhbq&dl=0" target="_blank">Dropbox</a> |
| vitb32_cifar10_w8a8_trim_60.pkl          | CIFAR-10 | ViT-B-32 (W8A8)  | <a href="https://www.dropbox.com/scl/fi/q25rxnli424b71fh23fmp/vitb32_cifar10_w8a8_trim_60.pkl?rlkey=lhg3t7lk7tw0owbcu655qy3sg&st=7zmant97&dl=0" target="_blank">Dropbox</a>         |
| vitb32_imagenet_w8a8_trim_40.pkl         | ImageNet | ViT-B-32 (W8A8)  | <a href="https://www.dropbox.com/scl/fi/ars6r7qr7ba736clmxl4x/vitb32_imagenet_w8a8_trim_40.pkl?rlkey=xfbopdu3a6mzgo5hz4abmkix3&st=ztobzv49&dl=0" target="_blank">Dropbox</a>        |

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@article{TODO,
  title={TODO},
  author={Zhang, Wenlun and Ando, Shimpei and Chen, Yung-Chin and Yoshioka, Kentaro},
  journal={TODO},
  year={2024}
}
