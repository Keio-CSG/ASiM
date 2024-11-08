"""

Some useful functions to be used in this project.

"""
import logging
import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms


def setup_seed(seed=6666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        # Accelerate training if training set does not change drastically. Best config for fixed cudnn (conv, etc.).


def inverse_transform(img_, transform_train):
    """
    Inverse transform dataset
    :param img_: Tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)     # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception('Invalid img shape, expected 1 or 3 in axis 2, but got {}!'.format(img_.shape[2]))

    return img_


def plot_result(train_x, train_y, valid_x, valid_y, mode, dir):
    """
    Plot loss/acc transfer curves for train & valid set.
    :param train_x: Epoch
    :param train_y: Scalar
    :param valid_x: Epoch
    :param valid_y: Scalar
    :param mode: 'loss' or 'acc'
    :param dir: Save figs to which dir
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(dir, mode + '.png'))
    plt.close()


def show_confmat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, figsize=None, perc=False):
    """
    Plot confusion matrix and save the figure
    :param confusion_mat: ndarray
    :param classes: list or tuple for class name
    :param set_name: str, dataset name, train or valid or test
    :param out_dir: str, directory to save figure
    :param epoch: int, @ which epoch
    :param verbose: bool, print information or not
    :param figsize: figure size
    :param perc: bool, show percentage
    :return:
    """
    class_num = len(classes)

    # Normalization
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # Fig size setup
    if class_num < 10:
        figsize = 6
    elif class_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[class_num-10]
    plt.figure(figsize=(int(figsize), int(figsize*1.3)))

    # Select color
    cmap = plt.cm.get_cmap('Greys')     # Color label: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_tmp, cmap=cmap)        # Color as percentage
    plt.colorbar(fraction=0.03)

    # Plot setup
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_{}_{}'.format(set_name, epoch))

    # Abs or %
    if perc:
        class_per_nums = confusion_mat.sum(axis=0)
        conf_mat_perc = confusion_mat / class_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s='{:.0%}'
                         .format(conf_mat_perc[i, j]), va='center', ha='center', color='red', fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)

    # Save fig
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_{}.png'.format(set_name)))
    plt.close()

    if verbose:
        for i in range(class_num):
            print('class: {:<10}, total num: {:<6}, correct num: {:<5} Recall: {:.2%} Precision: {:.2%}'
                  .format(classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                          confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                          confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))


def check_data_dir(dir):
    """
    Check directory.
    """
    assert os.path.exists(dir), \
        '\n\nDirectory does not exist. Current variable point to: \n{}\nPlease check relative directory settings or data files'.format(
        os.path.abspath(dir))


def make_logger(dir):
    """
    Create log file with input dir named with current time as recorder.
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%y-%m-%d_%H-%M')
    log_dir = os.path.join(dir, time_str)       # Create folder according to config
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Create logger
    path_log = os.path.join(log_dir, 'log.log')
    logger = Logger(path_log)
    logger = logger.init_logger()

    return logger, log_dir


class Logger():
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else 'root'
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # config file Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # config console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Add Handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


if __name__ == '__main__':      # Testbench
    setup_seed(6666)
    print(np.random.randint(0, 10, 1))

    logger = Logger('../resnet_cifar100/run/logtest.log')
    logger = logger.init_logger()
    for i in range(10):
        logger.info('test:' + str(i))

    from main.config import cfg
    logger.info(cfg)
