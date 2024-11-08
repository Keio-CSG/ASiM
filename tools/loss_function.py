"""

Customized loss functions.

"""
import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    # Step 1: Log Softmax
    # Step 2: Distribute weights. Real label = 1 - smoothing, the others = smoothing / (K - 1).
    # Step 3: Calculate loss by cross-entropy.

    def forward(self, input, target):
        log_p = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_p).sum(dim=-1).mean()

        return loss


if __name__ == '__main__':  # Testbench

    output = torch.tensor([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]])
    label = torch.tensor([2, 1, 1], dtype=torch.int64)

    loss_function = LabelSmoothLoss(0.001)
    loss = loss_function(output, label)

    print('CrossEntropy: {}.'.format(loss))