"""

Convert PyTorch official model to our .pkl checkpoint for pretraining.

"""

import torch


def convert_official_model_to_custom_format(official_model_path, output_model_path):
    # Load the official model weights
    official_weights = torch.load(official_model_path)

    # Create a new dictionary for the checkpoint in the format of checkpoint_best.pkl
    new_checkpoint = {
        'model_state_dict': official_weights,  # Use the official weights for model_state_dict
        'optimizer_state_dict': {},  # Empty optimizer state dict, assuming starting fresh
        'epoch': 0,  # Start from epoch 0
        'best_acc': 0.0  # Starting with 0.0 accuracy
    }

    # Save the new checkpoint
    torch.save(new_checkpoint, output_model_path)


if __name__ == "__main__":
    # Define paths
    official_model_path = r'E:\Machine_Learning\Project\ASiM\run\vit_b_16-c867db91.pth'
    output_model_path = r'E:\Machine_Learning\Project\ASiM\run\vit_b_16-c867db91.pkl'

    # Convert and save the weights
    convert_official_model_to_custom_format(official_model_path, output_model_path)
