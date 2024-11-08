"""

Dataset file of image classification task for <ImageNet Dataset>.

"""

import os
from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset with directory and preprocessing methods.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(dir, label), ... , ]
        self.class_num = 1000  # Number of categories
        self.names = tuple(range(self.class_num))
        self._get_img_info()

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at the specified index.
        """
        img_dir, label = self.img_info[index]
        img = Image.open(img_dir).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label, img_dir

    def __len__(self):
        """
        Get the total number of items in the dataset.
        """
        if not self.img_info:
            raise Exception(f'\ndata_dir: {self.root_dir} is an empty directory! Please check your directory settings!')
        return len(self.img_info)

    def _get_img_info(self):
        """
        Read dataset and summarize the directory & label in a list (img_info).
        [(dir, label), ... ,]
        """
        for label_num, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in sorted(os.listdir(class_dir)):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_dir, img_name)
                    self.img_info.append((img_path, label_num))


if __name__ == "__main__":      # Testbench

    # root_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train'
    root_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\valid'
    test_dataset = ImageNetDataset(root_dir)

    print(len(test_dataset))
    print(next(iter(test_dataset)))
