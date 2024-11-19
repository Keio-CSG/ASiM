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
| A6000 (48G)           | 1024       | ImageNet | ViT-B-32 (W8A8)  | 19841        |

## 4. ASiM

### 4.1. Abstract

<a href="https://arxiv.org/abs/2411.11022" target="_blank">Paper Link</a>

SRAM-based Analog Compute-in-Memory (ACiM) demonstrates promising energy efficiency for deep neural network (DNN) processing. Although recent aggressive design strategies have led to successive improvements on efficiency, there is limited discussion regarding the accompanying inference accuracy challenges. Given the growing difficulty in validating ACiM circuits with full-scale DNNs, standardized modeling methodology and open-source inference simulator are urgently needed. This paper presents ASiM, a simulation framework specifically designed to assess inference quality, enabling comparisons of ACiM prototype chips and guiding design decisions. ASiM works as a plug-and-play tool that integrates seamlessly with the PyTorch ecosystem, offering speed and ease of use. Using ASiM, we conducted a comprehensive analysis of how various design factors impact DNN inference. We observed that activation encoding can tolerate certain levels of quantization noise, indicating a substantial potential for bit-parallel scheme to enhance energy efficiency. However, inference accuracy is susceptible to noise, as ACiM circuits typically use limited ADC dynamic range, making even small errors down to 1 LSB significantly deteriorates accuracy. This underscores the need for high design standards, especially for complex DNN models and challenging tasks. In response to these findings, we propose two solutions: Hybrid Compute-in-Memory architecture and majority voting to secure accurate computation of MSB cycles. These approaches improve inference quality while maintaining ACiM’s energy efficiency benefits, offering promising pathways toward reliable ACiM deployment in real-world applications.

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

| File Name                                | Dataset  | Model            | Link                                                                                                                                     |
|------------------------------------------|----------|------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| resnet18_cifar10_w8a8_pact_trim_100.pkl  | CIFAR-10 | ResNet-18 (W8A8) | <a href="https://drive.google.com/file/d/1FoVjgLTTTZhFwD7plXZv1eWMHZzpEa-p/view?usp=drive_link" target="_blank">Google Drive</a>         |
| resnet18_imagenet_w8a8_pact_trim_50.pkl  | ImageNet | ResNet-18 (W8A8) | <a href="https://drive.google.com/file/d/1G42fgOUXJejV0f6rDMj2k3gDq1r7b3Z0/view?usp=drive_link" target="_blank">Google Drive</a>         |
| vitb32_cifar10_w8a8_trim_60.pkl          | CIFAR-10 | ViT-B-32 (W8A8)  | <a href="https://drive.google.com/file/d/1uypqLxoZA0hBX-UUOpk1-HsPAb3GDUqJ/view?usp=drive_link" target="_blank">Google Drive</a>         |
| vitb32_imagenet_w8a8_trim_40.pkl         | ImageNet | ViT-B-32 (W8A8)  | <a href="https://drive.google.com/file/d/1-3togKT0lIMjk4Bhsb5jO_DoqaNbUhGI/view?usp=drive_link" target="_blank">Google Drive</a>         |

## 5. Guiding Setup by Reproducing Results in the Paper

In this section, we will give the setup guide by reproducing some results reported in the paper. Please follow the steps and familiarize how ASiM works.

### 5.1. Prepare Environment and Datasets

Please refer our software versions provided in Sec. 2 for environment settings. Install easydict if you do not have it, and download our model weights provided in Sec. 4.3.

Download the following datasets: <a href="https://drive.google.com/file/d/1b7auKQzYE0VTe_0JCGCo8S6SzEl-WvuE/view?usp=drive_link" target="_blank">CIFAR-10</a>, <a href="https://drive.google.com/file/d/143wrVmpfqlH1Y0URvFVTFgyVhicO3ZA5/view?usp=drive_link" target="_blank">CIFAR-100</a>, and <a href="https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description" target="_blank">ImageNet</a>, respectively. Please use `./bins/valid_setup.py` to reformat the validation folder to match our dataset file of `./dataset/imagenet.py`.

The recollected dataset path should align with the following form:

CIFAR-10:

```
|-- CIFAR10
|   |-- train
|   |-- test
|   |-- labels.txt
```

CIFAR-100:

```
|-- CIFAR100
|   |-- train_images
|   |-- test_images
|   |-- train.csv
|   |-- test.csv
```

ImageNet:

```
|-- ImageNet
|   |-- train
|   |   |-- n01440764
|   |   |-- ...
|   |-- valid
|   |   |-- n01440764
|   |   |-- ...
```

### 5.2. Simulation Examples

#### 5.2.1. ResNet-18 on CIFAR-10

For simulations of ResNet-18 on CIFAR-10, please refer configuration at `./examples/resnet18_cifar10/config.py`. Please give the dataset path of validation set to `cfg.test_dir`, and pre-trained model weight to `cfg.pretrain_model_path`. Based on your devices, please set the appropriate batch size and work loader to `cfg.test_bs` and `cfg.test_workers`.

The default configuration is based on an 8b/8b model simulation on a bit-serial ACiM macro with row-parallelsim of 255. The CONV and Linear layer are configured separately. 

The model quantization bit can be modified by `wbit` and `xbit` parameters. The row-parallelsim and ADC precision can be modified by `adc_prec` and `nrow` parameters. For analog noise injection, please give `rand_noise_sigma` and `non_linear_sigma` based on your Silicon measurement. For bit-parallel scheme simulation, configure `act_enc` parameter with the encoding bit width (2, 4, etc.). To verify the efficacy of HCiM, configure the `cfg.asim_cnn_hybrid_levels` parameter with hybrid boundary level (1, 2, 3, etc.) to prevent MSB cycles from noise impact.

#### 5.2.2. ResNet-18 on ImageNet

Similarly, please set dataset/model weight path and refer `./examples/resnet18_imagenet/config.py` for configurations on ImageNet.

#### 5.2.3. ViT-B-32 on CIFAR-10

In Transformer models, the configuration of MultiheadAttention, MLP, and FC layers are given separately. Notably, in MultiheadAttention layer, we further give individual settings to QK, AV, and WQKV/WO projection, respectively. Please note the Keys (K) and Activations in projection layers are have a sign bit that have to be processed individually in bit-serial manner, we set `attn_qk_kbit` and `attn_proj_xbit` to 9 so that the LSBs can be evenly divided by 2 and 4, respectively. The HCiM boundary level in ViT is in `cfg.asim_vit_hybrid_levels`. Please refer `./examples/vitb32_cifar10/config.py` for more details.

#### 5.2.4. ViT-B-32 on ImageNet

Similarly, please set dataset/model weight path and refer `./examples/vitb32_imagenet/config.py` for configurations on ImageNet.

#### 5.2.5. Run Simulations

After the above configuration settings, just run `./main/src_simulation.py` for ASiM simulations.

### 5.3. Training Examples

We give an example of training ResNet-18 model on CIFAR-10 dataset for ACiM applications.

#### 5.3.1. Quantization-Aware Training (QAT)

For model training, the `cfg.train_dir` and `cfg.valid_dir` have to be set as your dataset root. To perform training without an pre-trained model, just give `cfg.pretrain_model_path` a **None**. If you wish to start by a pre-trained state, please give the directory of pre-trained model weight. Please note that you may have to rearrange the parameters from original PyTorch provided models, in these cases, the state_dict have to be reformed (refer scripts in `./bins`). 

Please set batch size and load workers by `cfg.train_bs`, `cfg.valid_bs`,  and `cfg.train_workers` respectively. Also, the `wbit` and `xbit` need to be configured to quantization weight and activation bit width, respectively.

#### 5.3.2. Noise-Aware Training (NAT)

For NAT, please give a pre-trained model to `cfg.pretrain_model_path`, and configure the noise intensity to `trim_noise` parameter.

#### 5.3.3. Train Models

After the above configuration settings, just run `./main/src_train.py` for model training.

Kindly remind that training conditions such as epoch/learning rate may need to be configured case by case (`cfg.max_epoch` and `cfg.lr_init`). A `run` directory will be generated when the training is finished that containing all training log and model weights.

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@article{ASiM,
  title={ASiM: Improving Transparency of SRAM-based Analog Compute-in-Memory Research with an Open-Source Simulation Framework},
  author={Zhang, Wenlun and Ando, Shimpei and Chen, Yung-Chin and Yoshioka, Kentaro},
  journal={arXiv:2411.11022},
  year={2024}
}
