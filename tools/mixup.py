"""

Mixup Implementation.

"""

import numpy as np
import torch


def mixup_data(data, label, alpha=1.0, device=True):
    """
    Returns mixed inputs, pairs of target, and lambda.
    """
    # Get lambda from beta distribution, where the parameter alpha == beta.
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    # To get labels from mixed images.
    batch_size = data.size()[0]
    index = torch.randperm(batch_size).to(device)

    # MixUp.
    mixed_img = lam * data + (1 - lam) * data[index, :]
    label_a, label_b = label, label[index]

    return mixed_img, label_a, label_b, lam


def mixup_loss_function(loss_function, pred, label_a, label_b, lamda):
    """
    Mixup integrated into loss function.
    """
    return lamda * loss_function(pred, label_a) + (1 - lamda) * loss_function(pred, label_b)


if __name__ == '__main__':  # Testbench
    import cv2
    import matplotlib.pyplot as plt
    path_1 = r'E:\Machine_Learning\Dataset\Processing\102flowers\jpg\image_00001.jpg'
    path_2 = r'E:\Machine_Learning\Dataset\Processing\102flowers\jpg\image_08061.jpg'

    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)
    img_1 = cv2.resize(img_1, (224, 224))
    img_2 = cv2.resize(img_2, (224, 224))

    alpha = 1.
    figsize = 15
    plt.figure(figsize=(int(figsize), int(figsize)))
    for i in range(1, 10):
        lamda = np.random.beta(alpha, alpha)
        img_mixup = (img_1 * lamda + img_2 * (1 - lamda)).astype(np.uint8)
        img_mixup = cv2.cvtColor(img_mixup, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 3, i)
        plt.title('lambda = {:.2f}'.format(lamda))
        plt.imshow(img_mixup)
    plt.show()
