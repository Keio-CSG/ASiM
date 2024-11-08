"""
Linear module for ASiM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tools.common_tools import setup_seed
from module.basic_module import Round, Clamp


class ASiMLinear(nn.Linear):
    """
    Bit-wise simulation Linear Module.
        in_features: input neurons
        out_features: output neurons
        bias: bias
        wbit: bit width of weight
        xbit: bit width of input activation
        adc_prec: ADC bit precision
        nrow: column length of macro (number of row parallelism)
        rand_noise_sigma: standard deviation of random noise in %
        non_linear_sigma: standard deviation of non-linear in %
        act_enc: activation encoding bit (bit-parallel)
        signed_act: signed or unsigned activations
        layer: layer type the module applied to, e.g.: 'fc' (fully-connected) or 'proj' (projection)
        hybrid_levels: msb cycle levels that perform in digital domain; default is None
        mode: operation mode, e.g.: 'Train' or 'Inference' or 'Simulation'
        trim_noise: standard deviation applied for noise-aware training in 'Train' mode or roughly evaluate noise impact in 'Inference' mode
        device: 'cpu' or 'cuda' or 'mps' depend on your device
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 wbit=4,
                 xbit=4,
                 adc_prec=8,
                 nrow=256,
                 rand_noise_sigma=0.0,
                 non_linear_sigma=0.0,
                 act_enc=None,
                 signed_act=True,
                 layer='fc',
                 hybrid_levels=None,
                 mode='Train',
                 trim_noise=0.0,
                 device='cpu'):
        super(ASiMLinear, self).__init__(in_features,
                                         out_features,
                                         bias)
        assert mode in ['Train', 'Inference', 'Simulation'], "Invalid mode specified. Choose from 'Train', 'Inference', or 'Simulation'."
        assert layer in ['fc', 'proj'], "Invalid layer specified. Choose from 'fc' or 'proj'."
        self.reset_parameters()
        self.in_channel = in_features
        self.out_channel = out_features
        self.wbit = wbit
        self.xbit = xbit
        self.adc_prec = adc_prec
        self.nrow = nrow
        self.rand_noise_sigma = rand_noise_sigma
        self.non_linear_sigma = non_linear_sigma
        self.bp = False
        self.signed_act = signed_act
        self.layer = layer
        self.bias_param = bias
        self.epsilon = 1e-7
        self.boundary_levels = 0
        self.mode = mode
        self.trim_noise = trim_noise / 100
        self.device = device
        if act_enc is not None:
            self.bp = True
            self.enc_bit = act_enc
            if signed_act is True:
                assert act_enc < xbit, "Signed activation: encoding should not exceed mantissa bit."
            else:
                assert act_enc <= xbit, "Unsigned activation: encoding should not exceed activation bit."
        if hybrid_levels is not None:
            self.boundary_levels = hybrid_levels

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
        return: decomposed INT result tensor with '0' and '1' in shape (wbit, out_channel, in_channel)
        """
        # Take sign bit (MSB) and calculate total value of remaining bits
        sign = torch.sign(input + self.epsilon).detach()
        sign = torch.abs((sign - 1) / 2)
        input = torch.abs(torch.abs(input) - sign * (2.0 ** (self.wbit - 1.0)))
        # Create tensor to store decomposed INT bit results
        w_map = torch.tensor([], device=self.device)
        # Integrate MSB
        w_map = torch.cat((w_map, sign))
        # Loop to integrate remaining bits
        for i in range(self.wbit-2, -1, -1):
            w_map = torch.cat((w_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
            input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (wbit, out_channel, in_channel), e.g.: (8, 100, 4096)
        w_map = w_map.reshape(self.wbit, self.out_channel, self.in_channel)
        return w_map

    def _decompose_feature(self, input):
        """
        Decompose FP32 input activation into quantized tensor with '0' and '1'.
        input: input activation tensor in FP32
        return: decomposed quantized result tensor with '0' and '1' in shape (xbit, batch, <seq_len>, in_channel)
        """
        batch = input.shape[0]
        if self.layer == 'proj':
            seq_len = input.shape[1]
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
        if self.layer == 'proj':
            # Reshape the tensor to (xbit, batch, seq_len, in_channel), e.g.: (8, 4, 64, 768)
            x_map = x_map.reshape(self.xbit, batch, seq_len, self.in_channel)
        else:
            # Reshape the tensor to (xbit, batch, in_channel), e.g.: (8, 4, 4096)
            x_map = x_map.reshape(self.xbit, batch, self.in_channel)
        return x_map

    def forward(self, input):
        """
        Forward call of ASiMLinear with selective operation mode.
        input: input activation tensor
        return: output activation tensor
        """
        if self.mode == 'Train':    # Training mode with noise-aware training option
            if self.signed_act:
                x_quant = self._quantize_signed_feature_train(input)
            else:
                x_quant = self._quantize_unsigned_feature_train(input)
            w_quant = self._quantize_weight_train(self.weight)
            if self.bias_param: # Compensate bias if needed
                output = F.linear(x_quant, w_quant, self.bias)
            else:   # No bias linear layer
                output = F.linear(x_quant, w_quant)
            # Add perturbation to output follows normal distribution with mean = 0 and standard deviation = trim_noise %
            perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.trim_noise * output
            output = output + perturb

        if self.mode == 'Inference':    # Inference mode that mimic quantized output with optional noise evaluation
            if self.signed_act:
                x_quant, x_scale = self._quantize_signed_feature_infer(input)
            else:
                x_quant, x_scale = self._quantize_unsigned_feature_infer(input)
            w_quant, w_scale = self._quantize_weight_infer(self.weight)
            # Compute linear without bias
            output = F.linear(x_quant, w_quant)
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
                output += self.bias

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
            xbit, batch = x_map.shape[0], x_map.shape[1]
            wbit = w_map.shape[0]
            # Initialize output activation tensors
            if self.layer == 'proj':
                seq_len = x_map.shape[-2]
                output = torch.zeros((batch, seq_len, self.out_channel), device=self.device)
            else:
                output = torch.zeros((batch, self.out_channel), device=self.device)
            # Calculate number of weight update
            w_update = math.ceil(self.in_channel / self.nrow)
            # Split bit map per weight update
            w_map = torch.split(w_map, self.nrow, dim=-1)
            x_map = torch.split(x_map, self.nrow, dim=-1)

            # Bit-parallel simulation
            if self.bp:
                # Signed activation branch
                if self.signed_act:
                    # Activation computing cycles after bit encoding
                    bp_cycle = math.ceil((self.xbit - 1) / self.enc_bit) + 1
                    # Calculate MSB encoding if there is remainder
                    msb_enc = (self.xbit - 1) % self.enc_bit
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
                                    # Computing activation sign bit using bit-wise linear
                                    bp_mac = F.linear(x_map[n][0].to(dtype=torch.float32),
                                                      w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                      bias=None)
                                    # Scale MAC results by full range and clamp MAC to [0, 1]
                                    bp_mac = torch.clamp(bp_mac / self.nrow, 0.0, 1.0)
                                    # Add noise to analog cycles
                                    if i + j <= wbit + bp_cycle - self.boundary_levels - 2:
                                        # Generate rand noise from given sigma%
                                        rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.rand_noise_sigma / 100
                                        # Generate non-linear from given sigma%
                                        nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.non_linear_sigma / 100
                                        # Add noise and clamp real MAC to [0, 1]
                                        bp_mac = bp_mac + rand_noise + nonl_noise
                                        bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                                    # Output MAC quantization by ADC
                                    bp_mac = torch.round(bp_mac * (2.0 ** self.adc_prec - 1.0))
                                    # Transform ADC results to bit-serial intermediate cycle result
                                    bp_mac = bp_mac / (2.0 ** self.adc_prec - 1.0) * self.nrow
                                    # Negative value for MSB weight
                                    if i != wbit - 1:
                                        bp_mac *= -1.0
                                    # Integrate bit-shifted partial sums into output activations
                                    output += torch.mul(bp_mac, 2.0 ** (i + xbit - 1.0))
                                    continue
                                if j == bp_cycle - 2:
                                    # Computing branch if the MSB encoding is different from enc_bit
                                    if msb_enc != 0:
                                        # Calculate ideal bit-parallel MAC result using bit-wise linear
                                        for k in range(msb_enc-1, -1, -1):
                                            bs_mac = F.linear(x_map[n][xbit-1-self.enc_bit*j-k].to(dtype=torch.float32),
                                                              w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                              bias=None)
                                            bp_mac += torch.mul(bs_mac, 2.0 ** k)
                                        # Scale MAC results by full range and clamp MAC to [0, 1]
                                        bp_mac = torch.clamp(bp_mac / ((2 ** msb_enc - 1) * self.nrow), 0.0, 1.0)
                                        # Add noise to analog cycles
                                        if i + j <= wbit + bp_cycle - self.boundary_levels - 2:
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
                                # Main computing branch for bit-parallel MAC result using bit-wise linear
                                for k in range(self.enc_bit-1, -1, -1):
                                    bs_mac = F.linear(x_map[n][xbit-1-self.enc_bit*j-k].to(dtype=torch.float32),
                                                      w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                      bias=None)
                                    bp_mac += torch.mul(bs_mac, 2.0 ** k)
                                # Scale MAC results by full range and clamp MAC to [0, 1]
                                bp_mac = torch.clamp(bp_mac / ((2 ** self.enc_bit - 1) * self.nrow), 0.0, 1.0)
                                # Add noise to analog cycles
                                if i + j <= wbit + bp_cycle - self.boundary_levels - 2:
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
                    output = output * (w_scale * x_scale) / ((2.0 ** (self.wbit - 1.0) - 1.0) * (2.0 ** (self.xbit - 1.0) - 1.0))
                # Unsigned activation branch
                else:
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
                                            bs_mac = F.linear(x_map[n][xbit-1-self.enc_bit*j-k].to(dtype=torch.float32),
                                                              w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                              bias=None)
                                            bp_mac += torch.mul(bs_mac, 2.0 ** k)
                                        # Scale MAC results by full range and clamp MAC to [0, 1]
                                        bp_mac = torch.clamp(bp_mac / ((2 ** msb_enc - 1) * self.nrow), 0.0, 1.0)
                                        # Add noise to analog cycles
                                        if i + j <= wbit + bp_cycle - self.boundary_levels - 2:
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
                                    bs_mac = F.linear(x_map[n][xbit-1-self.enc_bit*j-k].to(dtype=torch.float32),
                                                      w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                      bias=None)
                                    bp_mac += torch.mul(bs_mac, 2.0 ** k)
                                # Scale MAC results by full range and clamp MAC to [0, 1]
                                bp_mac = torch.clamp(bp_mac / ((2 ** self.enc_bit - 1) * self.nrow), 0.0, 1.0)
                                # Add noise to analog cycles
                                if i + j <= wbit + bp_cycle - self.boundary_levels - 2:
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
                            # Loop each activation bit to calculate ideal bit-serial MAC result using bit-wise linear
                            for j in range(xbit-1, -1, -1):
                                mac_col = F.linear(x_map[n][xbit-1-j].to(dtype=torch.float32),
                                                   w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                   bias=None)
                                # Scale MAC results by full range and clamp MAC to [0, 1]
                                mac_col = torch.clamp(mac_col / self.nrow, 0.0, 1.0)
                                # Add noise to analog cycles
                                if i + j <= wbit + xbit - self.boundary_levels - 2:
                                    # Generate rand noise from given sigma%
                                    rand_noise = torch.randn(mac_col.shape, device=self.device) * self.rand_noise_sigma / 100
                                    # Generate non-linear from given sigma%
                                    nonl_noise = torch.randn(mac_col.shape, device=self.device) / (torch.sqrt(mac_col * self.nrow + 1)) * self.non_linear_sigma / 100
                                    # Add noise and clamp real MAC to [0, 1]
                                    mac_col = mac_col + rand_noise + nonl_noise
                                    mac_col = torch.clamp(mac_col, 0.0, 1.0)
                                # Output MAC quantization by ADC
                                mac_col = torch.round(mac_col * (2.0 ** self.adc_prec - 1.0))
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
                    for n in range(w_update):
                        # Loop weight update
                        for i in range(wbit-1, -1, -1):
                            # Loop each activation bit to calculate ideal bit-serial MAC result using bit-wise linear
                            for j in range(xbit-1, -1, -1):
                                mac_col = F.linear(x_map[n][xbit-1-j].to(dtype=torch.float32),
                                                   w_map[n][wbit-1-i].to(dtype=torch.float32),
                                                   bias=None)
                                # Scale MAC results by full range and clamp MAC to [0, 1]
                                mac_col = torch.clamp(mac_col / self.nrow, 0.0, 1.0)
                                # Add noise to analog cycles
                                if i + j <= wbit + xbit - self.boundary_levels - 2:
                                    # Generate rand noise from given sigma%
                                    rand_noise = torch.randn(mac_col.shape, device=self.device) * self.rand_noise_sigma / 100
                                    # Generate non-linear from given sigma%
                                    nonl_noise = torch.randn(mac_col.shape, device=self.device) / (torch.sqrt(mac_col * self.nrow + 1)) * self.non_linear_sigma / 100
                                    # Add noise and clamp real MAC to [0, 1]
                                    mac_col = mac_col + rand_noise + nonl_noise
                                    mac_col = torch.clamp(mac_col, 0.0, 1.0)
                                # Output MAC quantization by ADC
                                mac_col = torch.round(mac_col * (2.0 ** self.adc_prec - 1.0))
                                # Transform ADC results to bit-serial intermediate cycle result
                                mac_col = mac_col / (2.0 ** self.adc_prec - 1.0) * self.nrow
                                # Negative value for MSB weight
                                if i == wbit - 1:
                                    mac_col *= -1.0
                                # Integrate bit-shifted partial sums into output activations
                                output += torch.mul(mac_col, 2.0 ** (i + j))
                    # Restore to FP32 scale
                    output = output * (w_scale * x_scale) / ((2.0 ** (self.wbit - 1.0) - 1.0) * (2.0 ** self.xbit - 1.0))
            # Compensation bias if needed
            if self.bias_param:
                output += self.bias

        return output


if __name__ == '__main__':  # testbench

    setup_seed(6666)
    device = 'cpu'

    # fake_linear_feature = torch.abs(torch.randn(4, 4096))
    fake_linear_feature = torch.randn(4, 4096)
    fake_linear_weight = torch.randn(100, 4096)
    fake_linear_bias = torch.randn(100)

    model_linear_train = ASiMLinear(4096,
                                    100,
                                    bias=True,
                                    wbit=4,
                                    xbit=4,
                                    adc_prec=100,
                                    nrow=256,
                                    rand_noise_sigma=0.0,
                                    non_linear_sigma=0.0,
                                    act_enc=None,
                                    signed_act=True,
                                    layer='fc',
                                    hybrid_levels=None,
                                    mode='Train',
                                    trim_noise=0.0,
                                    device='cpu')

    model_linear_infer = ASiMLinear(4096,
                                    100,
                                    bias=True,
                                    wbit=4,
                                    xbit=4,
                                    adc_prec=100,
                                    nrow=256,
                                    rand_noise_sigma=0.0,
                                    non_linear_sigma=0.0,
                                    act_enc=None,
                                    signed_act=True,
                                    layer='fc',
                                    hybrid_levels=None,
                                    mode='Inference',
                                    trim_noise=0.0,
                                    device='cpu')

    model_linear_sim = ASiMLinear(4096,
                                  100,
                                  bias=True,
                                  wbit=4,
                                  xbit=4,
                                  adc_prec=100,
                                  nrow=256,
                                  rand_noise_sigma=0.0,
                                  non_linear_sigma=0.0,
                                  act_enc=None,
                                  signed_act=True,
                                  layer='fc',
                                  hybrid_levels=None,
                                  mode='Simulation',
                                  trim_noise=0.0,
                                  device='cpu')

    model_linear_train._parameters['weight'] = fake_linear_weight
    model_linear_infer._parameters['weight'] = fake_linear_weight
    model_linear_sim._parameters['weight'] = fake_linear_weight
    model_linear_train._parameters['bias'] = fake_linear_bias
    model_linear_infer._parameters['bias'] = fake_linear_bias
    model_linear_sim._parameters['bias'] = fake_linear_bias

    output_linear_train = model_linear_train(fake_linear_feature)
    output_linear_infer = model_linear_infer(fake_linear_feature)
    output_linear_sim = model_linear_sim(fake_linear_feature)

    train_to_infer_linear_error = output_linear_infer - output_linear_train
    train_to_infer_linear_error_perc = train_to_infer_linear_error / output_linear_train
    infer_to_sim_linear_error = output_linear_sim - output_linear_infer
    infer_to_sim_linear_error_perc = infer_to_sim_linear_error / output_linear_infer

    print('Linear Layer: Train to Inference Error = {}, Sim to Inference Error = {}.'.format(train_to_infer_linear_error_perc, infer_to_sim_linear_error_perc))
