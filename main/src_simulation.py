"""

Main simulation code for the framework.

"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from main.config import cfg
from model.resnet_asim import resnet18_asim, resnet34_asim, resnet50_asim, resnet101_asim, resnet152_asim
from model.vgg_asim import vgg11_asim, vgg13_asim, vgg16_asim, vgg19_asim, vgg11_bn_asim, vgg13_bn_asim, vgg16_bn_asim, vgg19_bn_asim
from model.vision_transformer_asim import vit_b_16_asim, vit_b_32_asim, vit_l_16_asim, vit_l_32_asim, vit_h_14_asim
from dataset.cifar10 import Cifar10Dataset
from dataset.cifar100 import Cifar100Dataset
from dataset.imagenet import ImageNetDataset

if __name__ == '__main__': # Main script for simulation

    # Step 1: Dataset
    # Construct Dataset instance, and then construct DataLoader
    if cfg.task == 'CIFAR-10':
        test_data = Cifar10Dataset(root_dir=cfg.test_dir, transform=cfg.transforms_test)
    if cfg.task == 'CIFAR-100':
        test_data = Cifar100Dataset(root_dir=cfg.test_dir, transform=cfg.transforms_test)
    if cfg.task == 'ImageNet':
        test_data = ImageNetDataset(root_dir=cfg.test_dir, transform=cfg.transforms_test)

    test_loader = DataLoader(dataset=test_data, batch_size=cfg.test_bs, shuffle=False, num_workers=cfg.test_workers)

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
    pretrained_state_dict = torch.load(cfg.pretrain_model_path, map_location='cpu')
    model.load_state_dict(pretrained_state_dict['model_state_dict'])
    model.to(cfg.device)
    model.eval()

    # Step 3: Inference (Simulation)
    class_num = cfg.cls_num
    conf_mat = np.zeros((class_num, class_num))

    for i, data in enumerate(test_loader):
        inputs, labels, path_img = data
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

        outputs = model(inputs)

        # Generate confusion matrix
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.
            # Simulation print
            print('Label: {}, Predict: {}.'.format(cate_i, pre_i))

    acc_avg = conf_mat.trace() / conf_mat.sum()
    print('Test Acc: {:.2%}'.format(acc_avg))
