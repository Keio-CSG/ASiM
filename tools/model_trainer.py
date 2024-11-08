"""

Model Trainer.

"""

import torch
import numpy as np
from collections import Counter
from tools.mixup import mixup_data, mixup_loss_function


class ModelTrainer():

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, cfg, logger):
        model.train()

        class_num = data_loader.dataset.class_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        dir_misc = []
        label_list = []

        for i, data in enumerate(data_loader):

            # _, labels = data
            inputs, labels, img_dir = data
            label_list.extend(labels.tolist())

            # inputs, labels = data
            inputs, labels, img_dir = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup
            if cfg.mixup:
                mixed_inputs, label_a, label_b, lam = mixup_data(inputs, labels, cfg.mixup_alpha, device)
                inputs = mixed_inputs

            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()

            # Calculate loss
            if cfg.mixup:
                loss = mixup_loss_function(loss_f, outputs.cpu(), label_a.cpu(), label_b.cpu(), lam)
            else:
                loss = loss_f(outputs.cpu(), labels.cpu())

            loss.backward()
            optimizer.step()

            # Record loss
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)

            # Generate data for confusion matrix
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.
                if cate_i != pre_i:
                    dir_misc.append((cate_i, pre_i, img_dir[j]))        # Record information
            acc_avg = conf_mat.trace() / conf_mat.sum()     # Correct count on diagonal

            # Print training information every 10 interation.
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}'.
                            format(epoch_idx + 1, cfg.max_epoch, i + 1, len(data_loader), loss_mean, acc_avg))
        logger.info('Epoch: {} Sampler: {}'.format(epoch_idx + 1, Counter(label_list)))

        return loss_mean, acc_avg, conf_mat, dir_misc

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()

        class_num = data_loader.dataset.class_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        dir_misc = []

        for i, data in enumerate(data_loader):
            # inputs, labels = data
            inputs, labels, img_dir = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())

            # Generate data for confusion matrix
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.
                if cate_i != pre_i:
                    dir_misc.append((cate_i, pre_i, img_dir))     # Record information

            # Record loss
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()     # Correct count on the diagonal

        return np.mean(loss_sigma), acc_avg, conf_mat, dir_misc


if __name__ == '__main__':
    pass
