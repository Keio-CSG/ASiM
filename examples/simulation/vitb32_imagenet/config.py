"""

Parameter management config for training/simulation.

"""

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from easydict import EasyDict

# Obtain value via .key like key-value pair
cfg = EasyDict()

# <|Specify a task|>
# cfg.task = 'CIFAR-10'
# cfg.task = 'CIFAR-100'
cfg.task = 'ImageNet'

# <|Specify a model|>
# cfg.model_name = 'ResNet-18'
# cfg.model_name = 'ResNet-34'
# cfg.model_name = 'ResNet-50'
# cfg.model_name = 'ResNet-101'
# cfg.model_name = 'ResNet-152'
# cfg.model_name = 'VGG11'
# cfg.model_name = 'VGG13'
# cfg.model_name = 'VGG16'
# cfg.model_name = 'VGG19'
# cfg.model_name = 'VGG11-BN'
# cfg.model_name = 'VGG13-BN'
# cfg.model_name = 'VGG16-BN'
# cfg.model_name = 'VGG19-BN'
# cfg.model_name = 'ViT-B-16'
cfg.model_name = 'ViT-B-32'
# cfg.model_name = 'ViT-L-16'
# cfg.model_name = 'ViT-L-32'
# cfg.model_name = 'ViT-H-14'

# <|Specify the input image size|>
# CIFAR-10, CIFAR-100
# origin_size = 32
# input_size = 32
# ImageNet and ViT
origin_size = 256
input_size = 224

# <|Specify the running device: 'cpu', 'cuda', 'mps', etc.|>
cfg.device = 'cuda'

# <|Specify the classifier dimension|>
cfg.cls_num = 1000

#############################################################
################### <CNN module settings> ###################
#############################################################

# <|Specify using PACT or ReLU|>
cfg.PACT = True
# <|Specify Hybrid Computing Settings|>
cfg.asim_cnn_hybrid_levels = None # Hybrid computing boundary level

# <|Specify the settings for ASiMConv2d layers|>
cfg.asim_conv_mode = 'Simulation' # Mode: Train/Inference/Simulation
cfg.asim_conv_wbit = 8 # Weight bit
cfg.asim_conv_xbit = 8 # Activation bit
cfg.asim_conv_adc_prec = 8 # ADC precision
cfg.asim_conv_nrow = 255 # CiM column length (number of row-parallelism)
cfg.asim_conv_rand_noise_sigma = 0.0 # Standard deviation of random noise in %
cfg.asim_conv_non_linear_sigma = 0.0 # Standard deviation of non-linear in %
cfg.asim_act_enc = None # Activation encoding bit for bit-parallel computing (Give None for Bit-Serial Simulation)
cfg.asim_signed_act = False # Signed or unsigned activation
cfg.asim_conv_trim_noise = 0.0 # Noise intensity for noise-aware training or evaluate inference accuracy

# <|Specify the settings for ASiMLinear layers|>
cfg.asim_linear_mode = 'Simulation' # Mode: Train/Inference/Simulation
cfg.asim_linear_wbit = 8 # Weight bit
cfg.asim_linear_xbit = 8 # Activation bit
cfg.asim_linear_adc_prec = 8 # ADC precision
cfg.asim_linear_nrow = 255 # CiM column length (number of row-parallelism)
cfg.asim_linear_rand_noise_sigma = 0.0 # Standard deviation of random noise in %
cfg.asim_linear_non_linear_sigma = 0.0 # Standard deviation of non-linear in %
cfg.asim_linear_act_enc = None # Activation encoding bit for bit-parallel computing (Give None for Bit-Serial Simulation)
cfg.asim_linear_signed_act = False # Signed or unsigned activation
cfg.asim_linear_layer = 'fc' # Layer type ('fc' (fully-connected) or 'proj' (projection))
cfg.asim_linear_trim_noise = 0.0 # Noise intensity for noise-aware training or evaluate inference accuracy

# <|Specify the settings for QuantConv2d layers|>
cfg.quant_conv_mode = 'Train' # Mode: Train/Inference/Simulation
cfg.quant_conv_wbit = 8 # Weight bit
cfg.quant_conv_xbit = 8 # Activation bit

#############################################################
#############################################################
#############################################################

#############################################################
################### <ViT module settings> ###################
#############################################################

# <|Input image size|>
cfg.image_size = input_size
# <|If use QuantConv2d to extract token embedding|>
cfg.vit_quant_conv_proj = False
# <|Specify Hybrid Computing Settings|>
cfg.asim_vit_hybrid_levels = None # Hybrid computing boundary level

# <|Specify the settings for ASiMMultiheadAttention layers|>
cfg.asim_vit_attn_qk_qbit = 8 # Query bit in QK computation
cfg.asim_vit_attn_qk_kbit = 9 # Key bit in QK computation
cfg.asim_vit_attn_av_abit = 8 # Score bit in AV computation
cfg.asim_vit_attn_av_vbit = 8 # Value bit in AV computation
cfg.asim_vit_attn_proj_wbit = 8 # Weight bit in WX/WO projection
cfg.asim_vit_attn_proj_xbit = 9 # Activation bit in WX/WO projection
cfg.asim_vit_attn_qk_adc_prec = 8 # ADC precision in QK computation
cfg.asim_vit_attn_av_adc_prec = 8 # ADC precision in AV computation
cfg.asim_vit_attn_proj_adc_prec = 8 # ADC precision in WX/WO projection
cfg.asim_vit_attn_nrow = 255 # CiM column length (number of row-parallelism)
cfg.asim_vit_attn_qk_rand_noise_sigma = 0.0 # Standard deviation of random noise in % (QK computation)
cfg.asim_vit_attn_av_rand_noise_sigma = 0.0 # Standard deviation of random noise in % (AV computation)
cfg.asim_vit_attn_proj_rand_noise_sigma = 0.0 # Standard deviation of random noise in % (WQ/WK/WV/WO computation)
cfg.asim_vit_attn_qk_non_linear_sigma = 0.0 # Standard deviation of non-linear in % (QK computation)
cfg.asim_vit_attn_av_non_linear_sigma = 0.0 # Standard deviation of non-linear in % (AV computation)
cfg.asim_vit_attn_proj_non_linear_sigma = 0.0 # Standard deviation of non-linear in % (WQ/WK/WV/WO computation)
cfg.asim_vit_attn_qk_k_enc = None # Key encoding bit for bit-parallel computing (Give None for Bit-Serial Simulation)
cfg.asim_vit_attn_av_a_enc = None # Score encoding bit for bit-parallel computing (Give None for Bit-Serial Simulation)
cfg.asim_vit_attn_proj_act_enc = None # Activation encoding bit for bit-parallel computing (Give None for Bit-Serial Simulation)
cfg.asim_vit_attn_mode = 'Simulation' # Mode: Train/Inference/Simulation
cfg.asim_vit_attn_attn_trim_noise = 0.0 # Noise intensity for noise-aware training or evaluate inference accuracy (Attention)
cfg.asim_vit_attn_proj_trim_noise = 0.0 # Noise intensity for noise-aware training or evaluate inference accuracy (Projection)

# <|Specify the settings for MLP layers|>
cfg.asim_vit_mlp_wbit = 8 # Weight bit
cfg.asim_vit_mlp_xbit = 8 # Activation bit
cfg.asim_vit_mlp_adc_prec = 8 # ADC precision
cfg.asim_vit_mlp_nrow = 255 # CiM column length (number of row-parallelism)
cfg.asim_vit_mlp_rand_noise_sigma = 0.0 # Standard deviation of random noise in %
cfg.asim_vit_mlp_non_linear_sigma = 0.0 # Standard deviation of non-linear in %
cfg.asim_vit_mlp_act_enc = None # Activation encoding bit for bit-parallel computing (Give None for Bit-Serial Simulation)
cfg.asim_vit_mlp_mode = 'Simulation' # Mode: Train/Inference/Simulation
cfg.asim_vit_mlp_trim_noise = 0.0 # Noise intensity for noise-aware training or evaluate inference accuracy

# <|Specify the settings for FC layer|>
cfg.asim_vit_fc_wbit = 8 # Weight bit
cfg.asim_vit_fc_xbit = 8 # Activation bit
cfg.asim_vit_fc_adc_prec = 8 # ADC precision
cfg.asim_vit_fc_nrow = 255 # CiM column length (number of row-parallelism)
cfg.asim_vit_fc_rand_noise_sigma = 0.0 # Standard deviation of random noise in %
cfg.asim_vit_fc_non_linear_sigma = 0.0 # Standard deviation of non-linear in %
cfg.asim_vit_fc_act_enc = None # Activation encoding bit for bit-parallel computing (Give None for Bit-Serial Simulation)
cfg.asim_vit_fc_mode = 'Simulation' # Mode: Train/Inference/Simulation
cfg.asim_vit_fc_trim_noise = 0.0 # Noise intensity for noise-aware training or evaluate inference accuracy

# <|Specify the settings for QuantConv2d layers|>
cfg.asim_vit_quant_conv_wbit = 8 # Weight bit
cfg.asim_vit_quant_conv_xbit = 8 # Activation bit
cfg.asim_vit_quant_conv_mode = 'Train' # Mode: Train/Inference/Simulation

#############################################################
#############################################################
#############################################################

# <|Specify the directory for training/validation/test(simulation) dataset|>
# CIFAR-10
# cfg.train_dir = r'D:\Machine_Learning\Dataset\CIFAR10\train'
# cfg.valid_dir = r'D:\Machine_Learning\Dataset\CIFAR10\test'
# cfg.test_dir = r'D:\Machine_Learning\Dataset\CIFAR10\test'
# CIFAR-100
# cfg.train_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100\train_images'
# cfg.valid_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100\test_images'
# cfg.test_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100\test_images'
# ImageNet
cfg.train_dir = r'D:\Machine_Learning\Dataset\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train'
cfg.valid_dir = r'D:\Machine_Learning\Dataset\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\valid'
cfg.test_dir = r'D:\Machine_Learning\Dataset\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\valid'

# <|Specify the pretrained model for noise-aware training or test (simulation)|>
# cfg.pretrain_model_path = None  # For training, give a None.
cfg.pretrain_model_path = r'D:\Machine_Learning\ASiM_Models\vitb32_imagenet_w8a8_trim_40.pkl'

# <|Specify the batch size and workers for training|>
cfg.train_bs = 128
cfg.valid_bs = 128
cfg.train_workers = 8

# <|Specify the batch size and workers for test (simulation)|>
cfg.test_bs = 64
cfg.test_workers = 8

###############################################################
##################### <Training settings> #####################
###############################################################

# Settings for initial training
cfg.max_epoch = 200 # Training = 200, Fine-Tune = 80
cfg.lr_init = 0.01 # SGD Training = 0.01, SGD Fine-Tune = 0.001, Adam Training = 1e-4, Adam Fine-Tune = 1e-5
cfg.milestones = [int(cfg.max_epoch * 0.6), int(cfg.max_epoch * 0.9)]
cfg.mixup = False    # Use mixup or not.
cfg.label_smooth = False # Use label smoothing or not.

###############################################################
###############################################################
###############################################################


#####################################################################################################################
# No need to modify the following params in general
#####################################################################################################################

# <|Specify the optimizer parameters for training/noise-aware training|>
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1    # gamma (lr decay param) when using MultiStepLR
cfg.log_interval = 10 # Training log interval
cfg.mixup_alpha = 1.0    # Mixup parameter of beta distribution.
cfg.label_smooth_eps = 0.01 # Label smoothing eps parameter.

assert cfg.task in ['CIFAR-10', 'CIFAR-100', 'ImageNet'], "Invalid task specified. Choose from 'CIFAR-10', 'CIFAR-100', or 'ImageNet'."
if cfg.task == 'CIFAR-10':
    norm_mean = [0.4914, 0.4822, 0.4465]  # CIFAR-10
    norm_std = [0.2470, 0.2435, 0.2616]  # CIFAR-10
if cfg.task == 'CIFAR-100':
    norm_mean = [0.5071, 0.4867, 0.4408]    # CIFAR-100
    norm_std = [0.2675, 0.2565, 0.2761]     # CIFAR-100
if cfg.task == 'ImageNet':
    norm_mean = [0.485, 0.456, 0.406]  # ImageNet
    norm_std = [0.229, 0.224, 0.225]  # ImageNet

cfg.transforms_train = transforms.Compose([
    transforms.RandomChoice(
        [
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.3)
        ]
    ),
    transforms.Resize((origin_size, origin_size)),
    transforms.CenterCrop(origin_size),
    transforms.RandomCrop(input_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

cfg.transforms_valid = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

cfg.transforms_test = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

if input_size > 100:
    cfg.large_model = True
else:
    cfg.large_model = False

if __name__ == '__main__':      # Testbench

    from dataset.cifar100 import Cifar100Dataset
    from torch.utils.data import DataLoader
    from tools.common_tools import inverse_transform
    train_data = Cifar100Dataset(root_dir=cfg.train_dir, transform=cfg.transforms_train)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True)

    for epoch in range(cfg.max_epoch):
        for i, data in enumerate(train_loader):

            inputs, labels, dir = data       # B C H W

            img_tensor = inputs[0, ...]     # C H W
            img = inverse_transform(img_tensor, cfg.transforms_train)
            plt.imshow(img)
            plt.show()
            plt.pause(0.5)
            plt.close()
