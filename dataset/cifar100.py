"""

Dataset file of image classification task for <CIFAR-100 Dataset>.

"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Cifar100Dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset with directory and preprocessing methods.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(dir, label), ... , ]
        self.class_num = 100  # Number of categories
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
        """
        # List up image file paths.
        names_imgs = [os.path.join(self.root_dir, n) for n in os.listdir(self.root_dir) if n.endswith('.jpg')]

        # Read image IDs & labels from .csv files.
        train_label_file = os.path.join(self.root_dir, '..', 'train.csv')
        valid_label_file = os.path.join(self.root_dir, '..', 'test.csv')

        try:
            df_train = pd.read_csv(train_label_file)
            df_valid = pd.read_csv(valid_label_file)
        except FileNotFoundError as e:
            raise Exception(f'Error reading label files: {e}')

        df = pd.concat([df_train, df_valid])
        content = dict(zip(df['image_id'], df['fine_labels']))

        # Generate image file path and label pairs within the specified directory.
        self.img_info = [(p, content[os.path.basename(p)]) for p in names_imgs if os.path.basename(p) in content]


if __name__ == "__main__":      # Testbench

    root_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100\train_images'
    test_dataset = Cifar100Dataset(root_dir)

    print(len(test_dataset))
    print(next(iter(test_dataset)))
