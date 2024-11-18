"""

Convert PyTorch official model to our ASiM ResNet model checkpoint for pretraining.

"""

import sys
import torch
import torch.nn as nn
from module.basic_module import PACT
from module.asim_linear import ASiMLinear
from main.config import cfg
from model.resnet_asim import resnet18_asim, resnet34_asim, resnet50_asim, resnet101_asim, resnet152_asim


def transfer_weights(asim_model, official_weights):
    asim_state_dict = asim_model.state_dict()
    transferred_layers = 0
    total_layers = 0

    # Dictionary of weight block name in PyTorch model to be converted into ASiM model
    official_to_asim = {
        'bn1.weight': 'bn1.weight',
        'bn1.bias': 'bn1.bias',
        'bn1.running_mean': 'bn1.running_mean',
        'bn1.running_var': 'bn1.running_var',
        'fc.weight': 'fc.weight',
        'fc.bias': 'fc.bias',
    }

    # Map convolutional layer based on `cfg.large_model`
    if cfg.large_model:
        official_to_asim['conv1.weight'] = 'conv2.weight'  # Use conv2 if large_model is True
    else:
        official_to_asim['conv1.weight'] = 'conv1.weight'  # Use conv1 otherwise

    # Dynamically update dictionary for layer-by-layer mappings
    for i in range(4):
        for j in range(2):
            official_to_asim.update({
                f'layer{i+1}.{j}.conv1.weight': f'layer{i+1}.{j}.conv1.weight',
                f'layer{i+1}.{j}.bn1.weight': f'layer{i+1}.{j}.bn1.weight',
                f'layer{i+1}.{j}.bn1.bias': f'layer{i+1}.{j}.bn1.bias',
                f'layer{i+1}.{j}.bn1.running_mean': f'layer{i+1}.{j}.bn1.running_mean',
                f'layer{i+1}.{j}.bn1.running_var': f'layer{i+1}.{j}.bn1.running_var',
                f'layer{i+1}.{j}.conv2.weight': f'layer{i+1}.{j}.conv2.weight',
                f'layer{i+1}.{j}.bn2.weight': f'layer{i+1}.{j}.bn2.weight',
                f'layer{i+1}.{j}.bn2.bias': f'layer{i+1}.{j}.bn2.bias',
                f'layer{i+1}.{j}.bn2.running_mean': f'layer{i+1}.{j}.bn2.running_mean',
                f'layer{i+1}.{j}.bn2.running_var': f'layer{i+1}.{j}.bn2.running_var',
            })

            # Add downsample layers mapping if present
            if f'layer{i + 1}.{j}.downsample.0.weight' in asim_state_dict:
                official_to_asim.update({
                    f'layer{i + 1}.{j}.downsample.0.weight': f'layer{i + 1}.{j}.downsample.0.weight',
                    f'layer{i + 1}.{j}.downsample.1.running_mean': f'layer{i + 1}.{j}.downsample.1.running_mean',
                    f'layer{i + 1}.{j}.downsample.1.running_var': f'layer{i + 1}.{j}.downsample.1.running_var',
                    f'layer{i + 1}.{j}.downsample.1.weight': f'layer{i + 1}.{j}.downsample.1.weight',
                    f'layer{i + 1}.{j}.downsample.1.bias': f'layer{i + 1}.{j}.downsample.1.bias',
                })

    for name, param in official_weights.items():
        total_layers += 1
        if name in official_to_asim:
            asim_name = official_to_asim[name]
            if asim_name in asim_state_dict:
                try:
                    if asim_state_dict[asim_name].shape == param.shape:
                        asim_state_dict[asim_name].copy_(param)
                        transferred_layers += 1
                        print(f"Successfully copied weight for layer: {asim_name}")
                    else:
                        print(f"Shape mismatch for layer: {asim_name}, skipping...")
                except Exception as e:
                    print(f"Could not copy weight for layer: {asim_name}. Error: {e}")
            else:
                print(f"Layer {asim_name} not found in ASiM model.")
        else:
            print(f"Layer {name} not found in official to ASiM mapping.")

    if cfg.PACT:
        # Set default values for PACT
        for name, module in asim_model.named_modules():
            if isinstance(module, PACT):
                if 'alpha' not in module.__dict__:
                    module.alpha = nn.Parameter(torch.tensor(6.0))
                    print(f"Initialized alpha for {name}")

    asim_model.load_state_dict(asim_state_dict)
    print(f"Transferred {transferred_layers}/{total_layers} layers successfully.")
    sys.stdout.flush()


def modify_classifier_head(asim_model, num_classes):
    # Modify the classifier head to fit the class number
    in_features = asim_model.fc.in_features
    asim_model.fc = ASiMLinear(in_features, num_classes,
                               wbit=cfg.asim_linear_wbit,
                               xbit=cfg.asim_linear_xbit,
                               adc_prec=cfg.asim_linear_adc_prec,
                               nrow=cfg.asim_linear_nrow,
                               rand_noise_sigma=cfg.asim_conv_rand_noise_sigma,
                               non_linear_sigma=cfg.asim_conv_non_linear_sigma,
                               act_enc=cfg.asim_linear_act_enc,
                               signed_act=cfg.asim_linear_signed_act,
                               layer=cfg.asim_linear_layer,
                               mode=cfg.asim_linear_mode,
                               trim_noise=cfg.asim_linear_trim_noise,
                               device=cfg.device)
    nn.init.xavier_uniform_(asim_model.fc.weight)
    if asim_model.fc.bias is not None:
        nn.init.zeros_(asim_model.fc.bias)
    print(f"Modified classifier head to {num_classes} classes.")
    sys.stdout.flush()


def convert_official_model_to_asim_format(asim_model, num_classes, official_model_path, output_model_path):
    # Load the official model weights
    official_weights = torch.load(official_model_path)
    print(f"Loaded official model weights from {official_model_path}")
    sys.stdout.flush()

    # Transfer model weights
    transfer_weights(asim_model, official_weights)

    # Modify classifier head
    modify_classifier_head(asim_model, num_classes)

    # Create pickle checkpoint
    new_checkpoint = {
        'model_state_dict': asim_model.state_dict(),  # Converted ASiM model
        'optimizer_state_dict': {},  # Keep blank
        'epoch': 0,  # Keep blank
        'best_acc': 0.0  # Keep blank
    }

    # Save new checkpoint
    torch.save(new_checkpoint, output_model_path)
    print(f"Saved new model checkpoint to {output_model_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    # Define official model and output model directory
    official_model_path = r'E:\Machine_Learning\Model\resnet18-f37072fd.pth'  # Replace with your file path
    output_model_path = r'E:\Machine_Learning\Model\resnet18_cls_1000.pkl'

    # Instantiate ASiM model
    model = resnet18_asim()
    num_classes = 1000

    # Convert model
    convert_official_model_to_asim_format(model, num_classes, official_model_path, output_model_path)
