"""

Main training code for the framework.

"""

import sys
import pickle
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tools.model_trainer import ModelTrainer
from tools.common_tools import *
from main.config import cfg
from model.resnet_asim import resnet18_asim, resnet34_asim, resnet50_asim, resnet101_asim, resnet152_asim
from model.vgg_asim import vgg11_asim, vgg13_asim, vgg16_asim, vgg19_asim, vgg11_bn_asim, vgg13_bn_asim, vgg16_bn_asim, vgg19_bn_asim
from model.vision_transformer_asim import vit_b_16_asim, vit_b_32_asim, vit_l_16_asim, vit_l_32_asim, vit_h_14_asim
from datetime import datetime
from dataset.cifar10 import Cifar10Dataset
from dataset.cifar100 import Cifar100Dataset
from dataset.imagenet import ImageNetDataset
from tools.loss_function import LabelSmoothLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

if __name__ == '__main__': # Main script for model training

    # Step 0: Config
    # Generate Logger
    res_dir = os.path.join(BASE_DIR, 'run')
    logger, log_dir = make_logger(res_dir)

    # Step 1: Dataset
    # Construct Dataset instance, and then construct DataLoader
    if cfg.task == 'CIFAR-10':
        train_data = Cifar10Dataset(root_dir=cfg.train_dir, transform=cfg.transforms_train)
        valid_data = Cifar10Dataset(root_dir=cfg.valid_dir, transform=cfg.transforms_valid)
    if cfg.task == 'CIFAR-100':
        train_data = Cifar100Dataset(root_dir=cfg.train_dir, transform=cfg.transforms_train)
        valid_data = Cifar100Dataset(root_dir=cfg.valid_dir, transform=cfg.transforms_valid)
    if cfg.task == 'ImageNet':
        train_data = ImageNetDataset(root_dir=cfg.train_dir, transform=cfg.transforms_train)
        valid_data = ImageNetDataset(root_dir=cfg.valid_dir, transform=cfg.transforms_valid)

    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.train_workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.train_workers)

    # Step 2: Model Selection
    model_dic = {
        'ResNet-18': resnet18_asim(),
        'ResNet-34': resnet34_asim(),
        'ResNet-50': resnet50_asim(),
        'ResNet-101': resnet101_asim(),
        'ResNet-152': resnet152_asim(),
        'VGG11': vgg11_asim(),
        'VGG13': vgg13_asim(),
        'VGG16': vgg16_asim(),
        'VGG19': vgg19_asim(),
        'VGG11-BN': vgg11_bn_asim(),
        'VGG13-BN': vgg13_bn_asim(),
        'VGG16-BN': vgg16_bn_asim(),
        'VGG19-BN': vgg19_bn_asim(),
        'ViT-B-16': vit_b_16_asim(),
        'ViT-B-32': vit_b_32_asim(),
        'ViT-L-16': vit_l_16_asim(),
        'ViT-L-32': vit_l_32_asim(),
        'ViT-H-14': vit_h_14_asim()
    }
    model = model_dic[cfg.model_name]
    # Load pretrained model
    if cfg.pretrain_model_path is not None:
        pretrained_state_dict = torch.load(cfg.pretrain_model_path, map_location='cpu')
        model.load_state_dict(pretrained_state_dict['model_state_dict'])
        logger.info('Load pretrained model.')
    model.to(cfg.device)        # To device (cpu or gpu)

    # Step 3: Loss function & Optimizer
    if cfg.label_smooth:
        loss_f = LabelSmoothLoss(cfg.label_smooth_eps)
    else:
        loss_f = nn.CrossEntropyLoss()
    if cfg.model_name in ['ViT-B-16', 'ViT-B-32', 'ViT-L-16', 'ViT-L-32', 'ViT-H-14']:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr_init)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # Step 4: Training iteration
    # Record model, loss function, optimizer, and cfg in training
    logger.info('cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}\n'.format(
        cfg, loss_f, scheduler, optimizer, model))

    loss_rec = {'Train': [], 'Valid': []}
    acc_rec = {'Train': [], 'Valid': []}
    best_acc, best_epoch = 0, 0
    for epoch in range(cfg.max_epoch):

        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, cfg.device, cfg, logger)

        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, cfg.device)

        logger.info('Epoch[{:0>3}/{:0>3}], Train Acc: {:.2%}, Valid Acc: {:.2%}, Train Loss: {:.4f}, Valid Loss: {:.4f}, LR: {}'.format(
            epoch + 1, cfg.max_epoch, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]['lr']))
        scheduler.step()

        # Record training information
        loss_rec['Train'].append(loss_train), loss_rec['Valid'].append(loss_valid)
        acc_rec['Train'].append(acc_train), acc_rec['Valid'].append(acc_valid)
        # Confusion matrix
        show_confmat(mat_train, train_data.names, 'Train', log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        show_confmat(mat_valid, valid_data.names, 'Valid', log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        # Save loss, acc transfer curves
        plt_x = np.arange(1, epoch + 2)
        plot_result(plt_x, loss_rec['Train'], plt_x, loss_rec['Valid'], mode='loss', dir=log_dir)
        plot_result(plt_x, acc_rec['Train'], plt_x, acc_rec['Valid'], mode='acc', dir=log_dir)

        # Save model
        if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch,
                          'best_acc': best_acc}
            pkl_name = 'checkpoint_{}.pkl'.format(epoch + 1) if epoch == cfg.max_epoch - 1 else 'checkpoint_best.pkl'
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)

            # Save directory of misclassified figures
            misc_img_name = 'misc_img_{}.pkl'.format(epoch + 1) if epoch == cfg.max_epoch - 1 else 'misc_img_best.pkl'
            path_misc_img = os.path.join(log_dir, misc_img_name)
            misc_info = {}
            misc_info['Train'] = path_error_train
            misc_info['Valid'] = path_error_valid
            pickle.dump(misc_info, open(path_misc_img, 'wb'))

    logger.info('{} done, Best Acc: {} in epoch {}.'.format(
        datetime.strftime(datetime.now(), '%m-%d_%H_%M'), best_acc, best_epoch))
