"""

Settle validation set of ImageNet in the same form of training dataset.

"""

import os
import shutil
import xml.dom.minidom


def make_dir(dir):
    """
    Create directory if target root does not exist.
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)


if __name__ == "__main__":      # Testbench
    # 0. Config
    # Set abs directory of label file dir (xml file), newly created validation set dir, and val image dir.

    label_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Annotations\CLS-LOC\val'
    folder_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\valid'
    val_img_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\val'

    # 1. List up all label file dir.
    names_labels = os.listdir(label_dir)
    names_labels = [os.path.join(label_dir, n) for n in names_labels]

    # 2. Loop for copy files.
    for idx in names_labels:
        # Read filename and label name from corresponding xml file.
        dom = xml.dom.minidom.parse(idx)
        root = dom.documentElement
        target_file = dom.getElementsByTagName('filename')
        filename = target_file[0].firstChild.data
        target_label = dom.getElementsByTagName('name')
        label = target_label[0].firstChild.data
        # Set validation image abs dir and copy files.
        img_dir = os.path.join(val_img_dir, (filename + '.JPEG'))
        label_dir = os.path.join(folder_dir, label)
        make_dir(label_dir)
        shutil.copy(img_dir, label_dir)
        print('Copy {} imgs to {}'.format(filename, label))
