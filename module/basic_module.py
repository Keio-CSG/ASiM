"""
Basic classes and functions for models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.common_tools import setup_seed


class Floor(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input


class Round(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the round function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input


class Clamp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the clamp function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None, None


class PACTFunction(torch.autograd.Function):
    """
    Gradient computation function for PACT.
    """
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        return torch.clamp(input, min=0, max=alpha.item())

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_alpha = grad_output.clone()

        grad_input[input < 0] = 0
        grad_input[input > alpha] = 0

        grad_alpha = grad_alpha * ((input > alpha).float())

        return grad_input, grad_alpha.sum()


class PACT(nn.Module):
    """
    PACT ReLU Function
    """
    def __init__(self, alpha=6.0):
        super(PACT, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return PACTFunction.apply(x, self.alpha)


class QuantConv2d(nn.Conv2d):
    """
    Quantized Conv2d Module.
        in_channels: input channel
        out_channels: output channel
        kernel_size: kernel size
        stride: stride
        padding: padding
        dilation: dilation
        groups: groups
        bias: bias
        padding_mode: padding mode
        wbit: bit width of weight
        xbit: bit width of input activation
        signed_act: signed or unsigned activation
        mode: operation mode, e.g.: 'Train' or 'Inference'
        device: 'cpu' or 'cuda' or 'mps' depend on your device
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 wbit=8,
                 xbit=8,
                 signed_act=False,
                 mode='Train',
                 device='cpu'):
        super(QuantConv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          groups,
                                          bias,
                                          padding_mode)
        assert mode in ['Train', 'Inference'], "Invalid mode specified. Choose from 'Train', 'Inference'."
        self.wbit = wbit
        self.xbit = xbit
        self.signed_act = signed_act
        self.kernel_size_param = kernel_size
        self.in_channels_param = in_channels
        self.padding_param = padding
        self.stride_param = stride
        self.bias_param = bias
        self.epsilon = 1e-7
        self.mode = mode
        self.device = device

    def _quantize_weight_train(self, input):
        """
        Quantize weight tensor in 'Train' mode.
        input: input weight tensor
        return: fake quantized weight tensor for training
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        sign = torch.sign(input).detach()
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** (self.wbit - 1.0) - 1.0)) / (2.0 ** (self.wbit - 1.0) - 1.0) # INT Quantization
        return input * scaling * sign

    def _quantize_signed_feature_train(self, input):
        """
        Quantize signed input activation tensor in 'Train' mode.
        input: input activation tensor
        return: fake quantized input activation tensor for training
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        sign = torch.sign(input).detach()
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** (self.xbit - 1.0) - 1.0)) / (2.0 ** (self.xbit - 1.0) - 1.0)  # INT Quantization
        return input * scaling * sign

    def _quantize_unsigned_feature_train(self, input):
        """
        Quantize unsigned input activation tensor in 'Train' mode.
        input: input activation tensor
        return: fake quantized input activation tensor for training
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0)) / (2.0 ** self.xbit - 1.0)  # UINT Quantization
        return input * x_scale

    def _quantize_weight_infer(self, input):
        """
        Quantize weight tensor in 'Inference' mode.
        input: input weight tensor
        return: quantized weight with INT format for inference and scale factor for dequantization
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        sign = torch.sign(input).detach()
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** (self.wbit - 1.0) - 1.0))  # INT Quantization
        return input * sign, scaling

    def _quantize_signed_feature_infer(self, input):
        """
        Quantize signed input activation tensor in 'Inference' mode.
        input: input activation tensor
        return: quantized activation with INT format for inference and scale factor for dequantization
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        sign = torch.sign(input).detach()
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** (self.xbit - 1.0) - 1.0))  # INT Quantization
        return input * sign, scaling

    def _quantize_unsigned_feature_infer(self, input):
        """
        Quantize unsigned input activation tensor in 'Inference' mode.
        input: input activation tensor
        return: quantized activation with UINT format for inference and scale factor for dequantization
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0))  # UINT Quantization
        return input, x_scale

    def forward(self, input):
        """
        Forward call of QuantConv2d with selective operation mode.
        input: input activation tensor
        return: output activation tensor
        """
        if self.mode == 'Train':    # Training mode
            if self.signed_act:
                x_quant = self._quantize_signed_feature_train(input)
            else:
                x_quant = self._quantize_unsigned_feature_train(input)
            w_quant = self._quantize_weight_train(self.weight)
            output = F.conv2d(x_quant,
                              w_quant,
                              self.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)

        if self.mode == 'Inference':    # Inference mode that mimic quantized output
            if self.signed_act:
                x_quant, x_scale = self._quantize_signed_feature_infer(input)
            else:
                x_quant, x_scale = self._quantize_unsigned_feature_infer(input)
            w_quant, w_scale = self._quantize_weight_infer(self.weight)
            output = F.conv2d(x_quant,
                              w_quant,
                              bias=self.bias,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups)
            # De-quantization to FP32
            if self.signed_act:
                output = output * (w_scale * x_scale) / ((2.0 ** (self.wbit - 1.0) - 1.0) * (2.0 ** (self.xbit - 1.0) - 1.0))
            else:
                output = output * (w_scale * x_scale) / ((2.0 ** (self.wbit - 1.0) - 1.0) * (2.0 ** self.xbit - 1.0))
            # Compensation bias if needed
            if self.bias_param:
                x_height = output.shape[-2]
                x_width = output.shape[-1]
                bias_infer = self.bias.repeat(x_height, x_width, 1).permute(2, 0, 1)
                output += bias_infer

        return output


if __name__ == '__main__':  # testbench
    setup_seed(6666)
    device = 'cpu'

    # fake_conv_img = torch.abs(torch.randn((1, 64, 32, 32), device=device))
    fake_conv_img = torch.randn((1, 64, 32, 32), device=device)
    fake_conv_weight = torch.randn((1, 64, 3, 3), device=device)

    model_conv_train = QuantConv2d(64,
                                   1,
                                   kernel_size=3,
                                   stride=1,
                                   padding=0,
                                   bias=False,
                                   wbit=4,
                                   xbit=4,
                                   mode='Train',
                                   device='cpu')

    model_conv_infer = QuantConv2d(64,
                                   1,
                                   kernel_size=3,
                                   stride=1,
                                   padding=0,
                                   bias=False,
                                   wbit=4,
                                   xbit=4,
                                   mode='Inference',
                                   device='cpu')

    model_conv_train._parameters['weight'] = fake_conv_weight
    model_conv_infer._parameters['weight'] = fake_conv_weight

    output_conv_train = model_conv_train(fake_conv_img)
    output_conv_infer = model_conv_infer(fake_conv_img)

    train_to_infer_conv_error = output_conv_infer - output_conv_train
    train_to_infer_conv_error_perc = train_to_infer_conv_error / output_conv_train

    print('Conv Layer: Train to Inference Error = {}.'.format(train_to_infer_conv_error_perc))
