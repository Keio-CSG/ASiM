"""

Convert PyTorch official model to our ASiM ViT model checkpoint for pretraining.

"""

import sys
import torch
import torch.nn as nn
from main.config import cfg
from model.vision_transformer_asim import vit_b_16_asim, vit_b_32_asim, vit_l_16_asim, vit_l_32_asim, vit_h_14_asim
from module.asim_linear import ASiMLinear


def transfer_weights(asim_model, official_weights):
    asim_state_dict = asim_model.state_dict()
    transferred_layers = 0
    total_layers = 0

    # Dictionary of weight block name in PyTorch model to be converted into ASiM model
    official_to_asim = {
        'class_token': 'cls_token',
        'conv_proj.weight': 'patch_embedding.conv_proj.weight',
        'conv_proj.bias': 'patch_embedding.conv_proj.bias',
        'encoder.pos_embedding': 'encoder.pos_embedding',
        'heads.head.weight': 'heads.weight',
        'heads.head.bias': 'heads.bias',
        'encoder.ln.weight': 'encoder.ln.weight',
        'encoder.ln.bias': 'encoder.ln.bias',
    }

    for i in range(12):
        official_to_asim.update({
            f'encoder.layers.encoder_layer_{i}.ln_1.weight': f'encoder.layers.{i}.ln_1.weight',
            f'encoder.layers.encoder_layer_{i}.ln_1.bias': f'encoder.layers.{i}.ln_1.bias',
            f'encoder.layers.encoder_layer_{i}.self_attention.in_proj_weight': f'encoder.layers.{i}.self_attention.qkv.weight',
            f'encoder.layers.encoder_layer_{i}.self_attention.in_proj_bias': f'encoder.layers.{i}.self_attention.qkv.bias',
            f'encoder.layers.encoder_layer_{i}.self_attention.out_proj.weight': f'encoder.layers.{i}.self_attention.proj.weight',
            f'encoder.layers.encoder_layer_{i}.self_attention.out_proj.bias': f'encoder.layers.{i}.self_attention.proj.bias',
            f'encoder.layers.encoder_layer_{i}.ln_2.weight': f'encoder.layers.{i}.ln_2.weight',
            f'encoder.layers.encoder_layer_{i}.ln_2.bias': f'encoder.layers.{i}.ln_2.bias',
            f'encoder.layers.encoder_layer_{i}.mlp.linear_1.weight': f'encoder.layers.{i}.mlp.mlp1.weight',
            f'encoder.layers.encoder_layer_{i}.mlp.linear_1.bias': f'encoder.layers.{i}.mlp.mlp1.bias',
            f'encoder.layers.encoder_layer_{i}.mlp.linear_2.weight': f'encoder.layers.{i}.mlp.mlp2.weight',
            f'encoder.layers.encoder_layer_{i}.mlp.linear_2.bias': f'encoder.layers.{i}.mlp.mlp2.bias',
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

    asim_model.load_state_dict(asim_state_dict)
    print(f"Transferred {transferred_layers}/{total_layers} layers successfully.")
    sys.stdout.flush()


def modify_classifier_head(asim_model, num_classes):
    # Modify the classifier head to fit the class number
    in_features = asim_model.heads.in_features
    asim_model.heads = ASiMLinear(in_features,
                                  num_classes,
                                  bias=True,
                                  wbit=cfg.asim_vit_fc_wbit,
                                  xbit=cfg.asim_vit_fc_xbit,
                                  adc_prec=cfg.asim_vit_fc_adc_prec,
                                  nrow=cfg.asim_vit_fc_nrow,
                                  rand_noise_sigma=cfg.asim_vit_fc_rand_noise_sigma,
                                  non_linear_sigma=cfg.asim_vit_fc_non_linear_sigma,
                                  act_enc=cfg.asim_vit_fc_act_enc,
                                  signed_act=True,
                                  layer='fc',
                                  mode=cfg.asim_vit_fc_mode,
                                  trim_noise=cfg.asim_vit_fc_trim_noise,
                                  device=cfg.device)
    nn.init.xavier_uniform_(asim_model.heads.weight)
    if asim_model.heads.bias is not None:
        nn.init.zeros_(asim_model.heads.bias)
    print(f"Modified classifier head to {num_classes} classes.")
    sys.stdout.flush()


def convert_official_model_to_asim_format(asim_model, num_classes, official_model_path, output_model_path):
    # Load the official model weights
    official_weights = torch.load(official_model_path)
    print(f"Loaded official model weights from {official_model_path}")
    sys.stdout.flush()

    # Transfer model weights
    transfer_weights(asim_model, official_weights)

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
    official_model_path = r'E:\Machine_Learning\Model\vit_b_32-d86f8d99.pth'
    output_model_path = r'E:\Machine_Learning\Model\vit_b_32_cls_1000.pkl'

    # Give ASiM model
    model = vit_b_32_asim()
    num_classes = 1000

    # Convert model
    convert_official_model_to_asim_format(model, num_classes, official_model_path, output_model_path)
