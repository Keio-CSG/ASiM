"""

Draw test images (sampling) for simulation (CIFAR10/ImageNet).

"""

import os
import shutil
import random


def make_dir(dir):
    """
    Create directory if target root does not exist.
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)


if __name__ == "__main__":      # Testbench
    # 0. Config
    # Set random seed.
    # random_seed = 6666
    draw_count = 10000 # Number of images to draw from target path.

    # 1. Read list and shuffle.
    # Set root directory and data directory first.
    # random.seed(random_seed)
    root_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC'
    data_dir = os.path.join(root_dir, 'valid')
    # Create all subdirectories, mirroring the structure of data_dir
    for folder_name in os.listdir(data_dir):
        target_subdir = os.path.join(os.path.join(root_dir, 'images_under_test_10000'), folder_name)
        make_dir(target_subdir)  # This will create both empty and non-empty folders
    folder_dir = [p for p in os.listdir(data_dir)]
    folder_dir = [os.path.join(data_dir, name) for name in folder_dir]
    path_imgs = []
    for dir in folder_dir:
        names = [p for p in os.listdir(dir) if p.endswith('.JPEG')] # List up image names.
        paths = [os.path.join(dir, name) for name in names] # Create abs directory for all images.
        path_imgs += paths
    random.shuffle(path_imgs) # Shuffle images.

    # 2. Divide list by pre-defined ratio.
    draw_breakpoint = int(draw_count)
    test_imgs = path_imgs[:draw_breakpoint]

    test_folder = [p.split('\\')[-2] for p in test_imgs]
    target_dir = [os.path.join(root_dir, 'images_under_test_10000', n) for n in test_folder]

    # 3. Copy files.
    for draw in range(draw_count):
        make_dir(target_dir[draw])
        shutil.copy(test_imgs[draw], target_dir[draw])
        print('Copy {} to {}.'.format(test_imgs[draw], target_dir[draw]))
