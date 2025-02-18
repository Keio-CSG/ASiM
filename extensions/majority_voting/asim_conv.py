"""
CONV module for ASiM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tools.common_tools import setup_seed
from module.basic_module import Round, Clamp


class ASiMConv2d(nn.Conv2d):
    """
    ASiM Conv2d Module.
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
        adc_prec: ADC bit precision
        nrow: column length of macro (number of row parallelism)
        rand_noise_sigma: standard deviation of random noise in %
        non_linear_sigma: standard deviation of non-linear in %
        act_enc: activation encoding bit (bit-parallel)
        signed_act: signed or unsigned activations
        voting_levels: msb cycle levels that perform w/ majority voting; default is None
        num_samp: number of sampling times for majority voting
        mode: operation mode, e.g.: 'Train' or 'Inference' or 'Simulation'
        trim_noise: standard deviation applied for noise-aware training in 'Train' mode or roughly evaluate noise impact in 'Inference' mode
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
                 wbit=4,
                 xbit=4,
                 adc_prec=8,
                 nrow=256,
                 rand_noise_sigma=0.0,
                 non_linear_sigma=0.0,
                 act_enc=None,
                 signed_act=False,
                 voting_levels=None,
                 num_samp=None,
                 mode='Train',
                 trim_noise=0.0,
                 device='cpu'):
        super(ASiMConv2d, self).__init__(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups,
                                         bias,
                                         padding_mode)
        assert mode in ['Train', 'Inference', 'Simulation'], "Invalid mode specified. Choose from 'Train', 'Inference', or 'Simulation'."
        self.wbit = wbit
        self.xbit = xbit
        self.adc_prec = adc_prec
        self.nrow = nrow
        self.rand_noise_sigma = rand_noise_sigma
        self.non_linear_sigma = non_linear_sigma
        self.bp = False
        self.signed_act = signed_act
        self.kernel_size_param = kernel_size
        self.in_channels_param = in_channels
        self.padding_param = padding
        self.stride_param = stride
        self.bias_param = bias
        self.epsilon = 1e-7
        self.boundary_levels = 0
        self.num_samp = 1
        self.mode = mode
        self.trim_noise = trim_noise / 100
        self.device = device
        if act_enc is not None:
            self.bp = True
            self.enc_bit = act_enc
            assert act_enc <= xbit, "Activation encoding should not exceed activation bit."
            assert signed_act is False, "Cannot encode signed activation in bit parallel CiM."
        if voting_levels is not None:
            self.boundary_levels = voting_levels
        if num_samp is not None:
            assert act_enc is None, "Do not support oversampling with Bit-parallel mode in current version."
            self.num_samp = num_samp

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
        input = Round.apply(input * (2.0 ** (self.wbit - 1.0) - 1.0)) / (2.0 ** (self.wbit - 1.0) - 1.0)  # INT Quantization
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

    def _decompose_weight(self, input):
        """
        Decompose FP32 weight into INT quantized tensor with '0' and '1'.
        input: input weight tensor in FP32
        return: decomposed INT result tensor with '0' and '1' in shape (wbit, filter_num, channel, height, width)
        """
        filter_num, channel, height, width = input.shape
        # Take sign bit (MSB) and calculate total value of remaining bits
        sign = torch.sign(input + self.epsilon).detach()
        sign = torch.abs((sign - 1) / 2)
        input = torch.abs(torch.abs(input) - sign * (2.0 ** (self.wbit - 1.0)))
        # Create tensor to store decomposed INT bit results
        w_map = torch.tensor([], device=self.device)
        # Integrate sign bit
        w_map = torch.cat((w_map, sign))
        # Loop to integrate remaining bits
        for i in range(self.wbit-2, -1, -1):
            w_map = torch.cat((w_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
            input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (wbit, filter_num, channel, height, width), e.g.: (8, 128, 64, 32, 32)
        w_map = w_map.reshape(self.wbit, filter_num, channel, height, width)
        return w_map

    def _decompose_feature(self, input):
        """
        Decompose FP32 input activation into quantized tensor with '0' and '1'.
        input: input activation tensor in FP32
        return: decomposed quantized result tensor with '0' and '1' in shape (xbit, batch, channel, height, width)
        """
        batch, channel, height, width = input.shape
        if self.signed_act:
            # Take sign bit (MSB) and calculate total value of remaining bits
            sign = torch.sign(input + self.epsilon).detach()
            sign = torch.abs((sign - 1) / 2)
            input = torch.abs(torch.abs(input) - sign * (2.0 ** (self.xbit - 1.0)))
            # Create tensor to store decomposed INT bit results
            x_map = torch.tensor([], device=self.device)
            # Integrate MSB
            x_map = torch.cat((x_map, sign))
            # Loop to integrate remaining bits
            for i in range(self.xbit-2, -1, -1):
                x_map = torch.cat((x_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
                input = torch.remainder(input, 2.0 ** i)
        else:
            # Create tensor to store decomposed UINT bit results
            x_map = torch.tensor([], device=self.device)
            # Loop to integrate remaining bits
            for i in range(self.xbit-1, -1, -1):
                x_map = torch.cat((x_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
                input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (xbit, batch, channel, height, width), e.g.: (8, 4, 64, 32, 32)
        x_map = x_map.reshape(self.xbit, batch, channel, height, width)
        return x_map

    def forward(self, input):
        """
        Forward call of SimConv2d with selective operation mode.
        input: input activation tensor
        return: output activation tensor
        """
        if self.mode == 'Train':    # Training mode with noise-aware training option
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
            # Add perturbation to output follows normal distribution with mean = 0 and standard deviation = trim_noise %
            perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.trim_noise * output
            output = output + perturb

        if self.mode == 'Inference':    # Inference mode that mimic quantized output with optional noise evaluation
            if self.signed_act:
                x_quant, x_scale = self._quantize_signed_feature_infer(input)
            else:
                x_quant, x_scale = self._quantize_unsigned_feature_infer(input)
            w_quant, w_scale = self._quantize_weight_infer(self.weight)
            output = F.conv2d(x_quant,
                              w_quant,
                              bias=None,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups)
            # Add perturbation to output follows normal distribution with mean = 0 and standard deviation = trim_noise %
            perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.trim_noise * output
            output = output + perturb
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

        if self.mode == 'Simulation':   # ASiM simulation using the framework
            if self.signed_act:
                x_quant, x_scale = self._quantize_signed_feature_infer(input)
            else:
                x_quant, x_scale = self._quantize_unsigned_feature_infer(input)
            w_quant, w_scale = self._quantize_weight_infer(self.weight)
            # Decompose input bit map
            x_map = self._decompose_feature(x_quant).to(dtype=torch.int8)
            # Decompose weight bit map
            w_map = self._decompose_weight(w_quant).to(dtype=torch.int8)
            # Record dimension of weight and input activation tensor
            wbit, num_w, channel_w, height_w, width_w = w_map.shape
            xbit, batch_x, channel_x, height_x, width_x = x_map.shape
            # Calculate number of weight update
            w_update = math.ceil(self.in_channels_param * self.kernel_size_param * self.kernel_size_param / self.nrow)
            # Calculate channel dimension for macro mapping in each weight update: c/(w_update)
            channel_per_map = math.floor(self.nrow / (self.kernel_size_param * self.kernel_size_param))
            # Split bit map per weight update
            w_map = torch.split(w_map, channel_per_map, dim=2)
            x_map = torch.split(x_map, channel_per_map, dim=2)
            # Output activation tensor initialization
            height_o = math.floor((height_x - height_w + 2 * self.padding_param) / self.stride_param + 1)
            width_o = math.floor((width_x - width_w + 2 * self.padding_param) / self.stride_param + 1)
            output = torch.zeros(batch_x, num_w, height_o, width_o, device=self.device)
            # Bit-parallel simulation
            if self.bp:
                # Activation computing cycles after bit encoding
                bp_cycle = math.ceil(self.xbit / self.enc_bit)
                # Calculate MSB encoding if there is remainder
                msb_enc = self.xbit % self.enc_bit
                # Loop weight update
                for n in range(w_update):
                    # Loop each weight bit
                    for i in range(wbit-1, -1, -1):
                        # Loop each bit-parallel activation cycle
                        for j in range(bp_cycle-1, -1, -1):
                            # Initialize bit-parallel result tensor
                            bp_mac = torch.zeros_like(output, device=self.device)
                            # Bit-parallel cycle simulation
                            if j == bp_cycle - 1:
                                # Computing branch if the MSB encoding is different from enc_bit
                                if msb_enc != 0:
                                    # Calculate ideal bit-parallel MAC result using bit-wise conv2d
                                    for k in range(msb_enc-1, -1, -1):
                                        bs_mac = F.conv2d(
                                            x_map[n][xbit-1-self.enc_bit*j-k].to(dtype=torch.float32),
                                            w_map[n][wbit-1-i].to(dtype=torch.float32),
                                            bias=None,
                                            stride=self.stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            groups=self.groups)
                                        bp_mac += torch.mul(bs_mac, 2.0 ** k)
                                    # Scale MAC results by full range and clamp MAC to [0, 1]
                                    bp_mac = torch.clamp(bp_mac / ((2 ** msb_enc - 1) * self.nrow), 0.0, 1.0)
                                    # Generate rand noise from given sigma%
                                    rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.rand_noise_sigma / 100
                                    # Generate non-linear from given sigma%
                                    nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.non_linear_sigma / 100
                                    # Add noise and clamp real MAC to [0, 1]
                                    bp_mac = bp_mac + rand_noise + nonl_noise
                                    bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                                    # Output MAC quantization by ADC
                                    bp_mac = torch.round(bp_mac * (2.0 ** self.adc_prec - 1.0))
                                    # Transform ADC results to bit-parallel intermediate cycle result
                                    bp_mac = bp_mac / (2.0 ** self.adc_prec - 1.0) * ((2 ** msb_enc - 1) * self.nrow)
                                    # Negative value for MSB weight
                                    if i == wbit - 1:
                                        bp_mac *= -1.0
                                    # Integrate bit-shifted partial sums into output activations
                                    output += torch.mul(bp_mac, 2.0 ** (i + j * self.enc_bit))
                                    continue
                            # Main computing branch for bit-parallel MAC result using bit-wise conv2d
                            for k in range(self.enc_bit-1, -1, -1):
                                bs_mac = F.conv2d(x_map[n][xbit-1-self.enc_bit*j-k].to(dtype=torch.float32),
                                                  w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                  bias=None,
                                                  stride=self.stride,
                                                  padding=self.padding,
                                                  dilation=self.dilation,
                                                  groups=self.groups)
                                bp_mac += torch.mul(bs_mac, 2.0 ** k)
                            # Scale MAC results by full range and clamp MAC to [0, 1]
                            bp_mac = torch.clamp(bp_mac / ((2 ** self.enc_bit - 1) * self.nrow), 0.0, 1.0)
                            # Generate rand noise from given sigma%
                            rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.rand_noise_sigma / 100
                            # Generate non-linear from given sigma%
                            nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.non_linear_sigma / 100
                            # Add noise and clamp real MAC to [0, 1]
                            bp_mac = bp_mac + rand_noise + nonl_noise
                            bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                            # Output MAC quantization by ADC
                            bp_mac = torch.round(bp_mac * (2.0 ** self.adc_prec - 1.0))
                            # Transform ADC results to bit-parallel intermediate cycle result
                            bp_mac = bp_mac / (2.0 ** self.adc_prec - 1.0) * ((2 ** self.enc_bit - 1) * self.nrow)
                            # Negative value for MSB weight
                            if i == wbit - 1:
                                bp_mac *= -1.0
                            # Integrate bit-shifted partial sums into output activations
                            output += torch.mul(bp_mac, 2.0 ** (i + j * self.enc_bit))
                # Restore to FP32 scale
                output = output * (w_scale * x_scale) / ((2.0 ** (self.wbit - 1.0) - 1.0) * (2.0 ** self.xbit - 1.0))
            # Bit-serial simulation
            else:
                # Signed activation branch
                if self.signed_act:
                    # Loop weight update
                    for n in range(w_update):
                        # Loop each weight bit
                        for i in range(wbit-1, -1, -1):
                            # Loop each activation bit to calculate ideal bit-serial MAC result using bit-wise conv2d
                            for j in range(xbit-1, -1, -1):
                                # Oversampling
                                samp_results = []
                                if i + j > wbit + xbit - self.boundary_levels - 2:
                                    num_samp = self.num_samp
                                else:
                                    num_samp = 1
                                for _ in range(num_samp):
                                    mac_col = F.conv2d(x_map[n][xbit-1-j].to(dtype=torch.float32),
                                                       w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                       bias=None,
                                                       stride=self.stride,
                                                       padding=self.padding,
                                                       dilation=self.dilation,
                                                       groups=self.groups)
                                    # Scale MAC results by full range and clamp MAC to [0, 1]
                                    mac_col = torch.clamp(mac_col / self.nrow, 0.0, 1.0)
                                    # Generate rand noise from given sigma%
                                    rand_noise = torch.randn(mac_col.shape, device=self.device) * self.rand_noise_sigma / 100
                                    # Generate non-linear from given sigma%
                                    nonl_noise = torch.randn(mac_col.shape, device=self.device) / (torch.sqrt(mac_col * self.nrow + 1)) * self.non_linear_sigma / 100
                                    # Add noise and clamp real MAC to [0, 1]
                                    mac_col = mac_col + rand_noise + nonl_noise
                                    mac_col = torch.clamp(mac_col, 0.0, 1.0)
                                    # Output MAC quantization by ADC
                                    cur_mac_col = torch.round(mac_col * (2.0 ** self.adc_prec - 1.0))
                                    samp_results.append(cur_mac_col)
                                # Majority voting to get the most common result
                                mac_col = torch.mode(torch.stack(samp_results), dim=0).values
                                # Transform ADC results to bit-serial intermediate cycle result
                                mac_col = mac_col / (2.0 ** self.adc_prec - 1.0) * self.nrow
                                # Negative value for MSB weight and activation
                                if (i == (wbit - 1) or j == (xbit - 1)) and ((i + j) != (wbit + xbit - 2)):
                                    mac_col *= -1.0
                                # Integrate bit-shifted partial sums into output activations
                                output += torch.mul(mac_col, 2.0 ** (i + j))
                    # Restore to FP32 scale
                    output = output * (w_scale * x_scale) / ((2.0 ** (self.wbit - 1.0) - 1.0) * (2.0 ** (self.xbit - 1.0) - 1.0))
                else:
                    # Unsigned activation branch
                    # Loop weight update
                    for n in range(w_update):
                        # Loop each weight bit
                        for i in range(wbit-1, -1, -1):
                            # Loop each activation bit to calculate ideal bit-serial MAC result using bit-wise conv2d
                            for j in range(xbit-1, -1, -1):
                                # Oversampling
                                samp_results = []
                                if i + j > wbit + xbit - self.boundary_levels - 2:
                                    num_samp = self.num_samp
                                else:
                                    num_samp = 1
                                for _ in range(num_samp):
                                    mac_col = F.conv2d(x_map[n][xbit - 1 - j].to(dtype=torch.float32),
                                                       w_map[n][wbit - 1 - i].to(dtype=torch.float32),
                                                       bias=None,
                                                       stride=self.stride,
                                                       padding=self.padding,
                                                       dilation=self.dilation,
                                                       groups=self.groups)
                                    # Scale MAC results by full range and clamp MAC to [0, 1]
                                    mac_col = torch.clamp(mac_col / self.nrow, 0.0, 1.0)
                                    # Generate rand noise from given sigma%
                                    rand_noise = torch.randn(mac_col.shape, device=self.device) * self.rand_noise_sigma / 100
                                    # Generate non-linear from given sigma%
                                    nonl_noise = torch.randn(mac_col.shape, device=self.device) / (torch.sqrt(mac_col * self.nrow + 1)) * self.non_linear_sigma / 100
                                    # Add noise and clamp real MAC to [0, 1]
                                    mac_col = mac_col + rand_noise + nonl_noise
                                    mac_col = torch.clamp(mac_col, 0.0, 1.0)
                                    # Output MAC quantization by ADC
                                    cur_mac_col = torch.round(mac_col * (2.0 ** self.adc_prec - 1.0))
                                    samp_results.append(cur_mac_col)
                                # Majority voting to get the most common result
                                mac_col = torch.mode(torch.stack(samp_results), dim=0).values
                                # Transform ADC results to bit-serial intermediate cycle result
                                mac_col = mac_col / (2.0 ** self.adc_prec - 1.0) * self.nrow
                                # Negative value for MSB weight
                                if i == wbit-1:
                                    mac_col *= -1.0
                                # Integrate bit-shifted partial sums into output activations
                                output += torch.mul(mac_col, 2.0 ** (i + j))
                    # Restore to FP32 scale
                    output = output * (w_scale * x_scale) / ((2.0 ** (self.wbit - 1.0) - 1.0) * (2.0 ** self.xbit - 1.0))
            # Compensation bias if needed
            if self.bias_param:
                x_height = output.shape[-2]
                x_width = output.shape[-1]
                bias_sim = self.bias.repeat(x_height, x_width, 1).permute(2, 0, 1)
                output += bias_sim

        return output


if __name__ == '__main__':  # testbench

    setup_seed(6666)
    device = 'cpu'

    # fake_conv_img = torch.abs(torch.randn((1, 64, 32, 32), device=device))
    fake_conv_img = torch.randn((1, 64, 32, 32), device=device)
    fake_conv_weight = torch.randn((1, 64, 3, 3), device=device)

    model_conv_train = ASiMConv2d(64,
                                  1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=0,
                                  bias=False,
                                  wbit=4,
                                  xbit=4,
                                  adc_prec=100,
                                  nrow=256,
                                  rand_noise_sigma=0.0,
                                  non_linear_sigma=0.0,
                                  act_enc=None,
                                  signed_act=True,
                                  voting_levels=None,
                                  num_samp=None,
                                  mode='Train',
                                  trim_noise=0.0,
                                  device='cpu')

    model_conv_infer = ASiMConv2d(64,
                                  1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=0,
                                  bias=False,
                                  wbit=4,
                                  xbit=4,
                                  adc_prec=100,
                                  nrow=256,
                                  rand_noise_sigma=0.0,
                                  non_linear_sigma=0.0,
                                  act_enc=None,
                                  signed_act=True,
                                  voting_levels=None,
                                  num_samp=None,
                                  mode='Inference',
                                  trim_noise=0.0,
                                  device='cpu')

    model_conv_sim = ASiMConv2d(64,
                                1,
                                kernel_size=3,
                                stride=1,
                                padding=0,
                                bias=False,
                                wbit=4,
                                xbit=4,
                                adc_prec=100,
                                nrow=256,
                                rand_noise_sigma=0.0,
                                non_linear_sigma=0.0,
                                act_enc=None,
                                signed_act=True,
                                voting_levels=None,
                                num_samp=None,
                                mode='Simulation',
                                trim_noise=0.0,
                                device='cpu')

    model_conv_train._parameters['weight'] = fake_conv_weight
    model_conv_infer._parameters['weight'] = fake_conv_weight
    model_conv_sim._parameters['weight'] = fake_conv_weight

    output_conv_train = model_conv_train(fake_conv_img)
    output_conv_infer = model_conv_infer(fake_conv_img)
    output_conv_sim = model_conv_sim(fake_conv_img)

    train_to_infer_conv_error = output_conv_infer - output_conv_train
    train_to_infer_conv_error_perc = train_to_infer_conv_error / output_conv_train
    infer_to_sim_conv_error = output_conv_sim - output_conv_infer
    infer_to_sim_conv_error_perc = infer_to_sim_conv_error / output_conv_infer

    print('Conv Layer: Train to Inference Error = {}, Sim to Inference Error = {}.'.format(train_to_infer_conv_error_perc, infer_to_sim_conv_error_perc))
