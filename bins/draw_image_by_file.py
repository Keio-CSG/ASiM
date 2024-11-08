"""

Draw test images (sampling) for simulation (CIFAR-100).

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


def copy_img(imgs, root_dir, setname):
    """
    Copy image files from original directory to target directory.
    :param imgs: Original images abs directory.
    :param root_dir: Target root directory.
    :param setname: Target set directory name.
    """
    data_dir = os.path.join(root_dir, setname)
    make_dir(data_dir)
    for path_img in imgs:
        print(path_img)
        shutil.copy(path_img, data_dir)
    print('{} dataset, copy {} imgs to {}'.format(setname, len(imgs), data_dir))


if __name__ == "__main__":
    # 0. Config
    # random_seed = 6666
    draw_ratio = 0.01 # Ratio to draw from target path.

    # 1. Read list and shuffle.
    # Set root directory and data directory first.
    # random.seed(random_seed)
    root_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100'
    data_dir = os.path.join(root_dir, 'test_images')
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith('.jpg')]         # List up image names.
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs]            # Create abs directory for all images.
    random.shuffle(path_imgs)                                                   # Shuffle images.

    # 2. Divide list into 3 by pre-defined ratio.
    draw_breakpoint = int(len(path_imgs) * draw_ratio)
    test_imgs = path_imgs[:draw_breakpoint]

    # 3. Copy files.
    copy_img(test_imgs, root_dir, 'images_under_test_100')
