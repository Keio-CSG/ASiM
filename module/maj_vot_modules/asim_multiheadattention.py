"""
MultiheadAttention module for ASiM.
"""

import torch
import torch.nn as nn
import math
from module.basic_module import Round, Clamp
from module.asim_linear import ASiMLinear


class ASiMMultiheadAttention(nn.Module):
    """
    Bit-wise simulation MultiheadAttention Module.
        embed_dim: embedding dimension
        num_heads: number of heads
        dropout: dropout ratio
        qk_qbit: query bit width of QK
        qk_kbit: key bit width of QK
        av_abit: attention bit width of AV
        av_vbit: value bit width of AV
        proj_wbit: weight bit width of projection layer
        proj_xbit: activation bit width of projection layer
        qk_adc_prec: ADC bit precision of QK computing
        av_adc_prec: ADC bit precision of AV computing
        proj_adc_prec: ADC bit precision of projection layer
        nrow: column length of macro (number of row parallelism)
        qk_rand_noise_sigma: standard deviation of random noise in % in QK computing
        av_rand_noise_sigma: standard deviation of random noise in % in AV computing
        proj_rand_noise_sigma: standard deviation of random noise in % in projection layer computing
        qk_non_linear_sigma: standard deviation of non-linear in % in QK computing
        av_non_linear_sigma: standard deviation of non-linear in % in AV computing
        proj_non_linear_sigma: standard deviation of non-linear in % in projection layer computing
        qk_k_enc: K encoding bit (bit-parallel)
        av_a_enc: A encoding bit (bit-parallel)
        proj_act_enc: activation encoding in projection layer (bit-parallel)
        voting_levels: msb cycle levels that perform w/ majority voting; default is None
        num_samp: number of sampling times for majority voting
        mode: operation mode, e.g.: 'Train' or 'Inference' or 'Simulation'
        attn_trim_noise: standard deviation applied for noise-aware training in 'Train' mode or roughly evaluate noise impact in 'Inference' mode (QK/AV)
        proj_trim_noise: standard deviation applied for noise-aware training in 'Train' mode or roughly evaluate noise impact in 'Inference' mode (projection)
        device: 'cpu' or 'cuda' or 'mps' depend on your device
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 qk_qbit=8,
                 qk_kbit=8,
                 av_abit=8,
                 av_vbit=8,
                 proj_wbit=8,
                 proj_xbit=8,
                 qk_adc_prec=8,
                 av_adc_prec=8,
                 proj_adc_prec=8,
                 nrow=256,
                 qk_rand_noise_sigma=0.0,
                 av_rand_noise_sigma=0.0,
                 proj_rand_noise_sigma=0.0,
                 qk_non_linear_sigma=0.0,
                 av_non_linear_sigma=0.0,
                 proj_non_linear_sigma=0.0,
                 qk_k_enc=None,
                 av_a_enc=None,
                 proj_act_enc=None,
                 voting_levels=None,
                 num_samp=None,
                 mode='Train',
                 attn_trim_noise=0.0,
                 proj_trim_noise=0.0,
                 device='cpu'):
        super().__init__()
        assert mode in ['Train', 'Inference', 'Simulation'], "Invalid mode specified. Choose from 'Train', 'Inference', or 'Simulation'."
        self.num_heads = num_heads
        self.scale = embed_dim ** -0.5
        self.qk_qbit = qk_qbit
        self.qk_kbit = qk_kbit
        self.av_abit = av_abit
        self.av_vbit = av_vbit
        self.proj_wbit = proj_wbit
        self.proj_xbit = proj_xbit
        self.qk_adc_prec = qk_adc_prec
        self.av_adc_prec = av_adc_prec
        self.proj_adc_prec = proj_adc_prec
        self.nrow = nrow
        self.qk_rand_noise_sigma = qk_rand_noise_sigma
        self.av_rand_noise_sigma = av_rand_noise_sigma
        self.proj_rand_noise_sigma = proj_rand_noise_sigma
        self.qk_non_linear_sigma = qk_non_linear_sigma
        self.av_non_linear_sigma = av_non_linear_sigma
        self.proj_non_linear_sigma = proj_non_linear_sigma
        self.qk_bp = False
        self.av_bp = False
        self.proj_bp = False
        self.boundary_levels = 0
        self.num_samp = 1
        self.mode = mode
        self.attn_trim_noise = attn_trim_noise / 100
        self.proj_trim_noise = proj_trim_noise / 100
        self.device = device
        self.epsilon = 1e-7
        self.dropout_layer = nn.Dropout(dropout)
        self.qkv = ASiMLinear(embed_dim,
                              embed_dim * 3,
                              wbit=proj_wbit,
                              xbit=proj_xbit,
                              adc_prec=proj_adc_prec,
                              nrow=nrow,
                              rand_noise_sigma=proj_rand_noise_sigma,
                              non_linear_sigma=proj_non_linear_sigma,
                              act_enc=proj_act_enc,
                              signed_act=True,
                              layer='proj',
                              mode=mode,
                              voting_levels=voting_levels,
                              num_samp=num_samp,
                              trim_noise=proj_trim_noise,
                              device=device)
        self.proj = ASiMLinear(embed_dim,
                               embed_dim,
                               wbit=proj_wbit,
                               xbit=proj_xbit,
                               adc_prec=proj_adc_prec,
                               nrow=nrow,
                               rand_noise_sigma=proj_rand_noise_sigma,
                               non_linear_sigma=proj_non_linear_sigma,
                               act_enc=proj_act_enc,
                               signed_act=True,
                               layer='proj',
                               mode=mode,
                               voting_levels=voting_levels,
                               num_samp=num_samp,
                               trim_noise=proj_trim_noise,
                               device=device)
        if qk_k_enc is not None:
            self.qk_bp = True
            self.qk_k_enc = qk_k_enc
            assert qk_k_enc < qk_qbit, "Attention Key: encoding should not exceed mantissa bit."
        if av_a_enc is not None:
            self.av_bp = True
            self.av_a_enc = av_a_enc
            assert av_a_enc <= av_abit, "Attention Score: encoding should not exceed quantization bit."
        if voting_levels is not None:
            self.boundary_levels = voting_levels
        if num_samp is not None:
            assert all(enc is None for enc in [qk_k_enc, av_a_enc, proj_act_enc]), "Do not support oversampling with Bit-parallel mode in current version."
            self.num_samp = num_samp

    def _quantize_signed_matrix_train(self, input, bit):
        """
        Quantize signed input matrix tensor in 'Train' mode.
        input: input matrix tensor, quantized bit
        return: fake quantized input matrix tensor for training
        """
        assert bit > 1, "Bit width must be greater than 1."
        sign = torch.sign(input).detach()
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** (bit - 1.0) - 1.0)) / (2.0 ** (bit - 1.0) - 1.0)  # INT Quantization
        return input * scaling * sign

    def _quantize_unsigned_matrix_train(self, input, bit):
        """
        Quantize unsigned input matrix tensor in 'Train' mode.
        input: input matrix tensor, quantized bit
        return: fake quantized input matrix tensor for training
        """
        assert bit > 1, "Bit width must be greater than 1."
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** bit - 1.0)) / (2.0 ** bit - 1.0)  # UINT Quantization
        return input * scaling

    def _quantize_signed_matrix_infer(self, input, bit):
        """
        Quantize signed input matrix tensor in 'Inference' mode.
        input: input matrix tensor, quantized bit
        return: fake quantized input matrix tensor and scale factor for inference
        """
        assert bit > 1, "Bit width must be greater than 1."
        sign = torch.sign(input).detach()
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** (bit - 1.0) - 1.0))  # INT Quantization
        return input * sign, scaling

    def _quantize_unsigned_matrix_infer(self, input, bit):
        """
        Quantize unsigned input matrix tensor in 'Inference' mode.
        input: input matrix tensor, quantized bit
        return: fake quantized input matrix tensor and scale factor for inference
        """
        assert bit > 1, "Bit width must be greater than 1."
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / scaling, 0.0, 1.0)  # range from 0~1
        input = Round.apply(input * (2.0 ** bit - 1.0))  # UINT Quantization
        return input, scaling

    def _decompose_matrix(self, input, bit, signed):
        """
        Decompose FP32 input activation into quantized tensor with '0' and '1'.
        input: input activation tensor in FP32, quantized bit, signed or unsigned input
        return: decomposed quantized result tensor with '0' and '1' in shape (bit, batch, head, m_dim, n_dim)
        """
        batch, head, m, n = input.shape
        if signed:
            # Take sign bit (MSB) and calculate total value of remaining bits
            sign = torch.sign(input + self.epsilon).detach()
            sign = torch.abs((sign - 1) / 2)
            input = torch.abs(torch.abs(input) - sign * (2.0 ** (bit - 1.0)))
            # Create tensor to store decomposed INT bit results
            bit_map = torch.tensor([], device=self.device)
            # Integrate MSB
            bit_map = torch.cat((bit_map, sign))
            # Loop to integrate remaining bits
            for i in range(bit-2, -1, -1):
                bit_map = torch.cat((bit_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
                input = torch.remainder(input, 2.0 ** i)
        else:
            # Create tensor to store decomposed UINT bit results
            bit_map = torch.tensor([], device=self.device)
            # Loop to integrate remaining bits
            for i in range(bit-1, -1, -1):
                bit_map = torch.cat((bit_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
                input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (bit, batch, head, m, n), e.g.: (8, 64, 8, 64, 768)
        bit_map = bit_map.reshape(bit, batch, head, m, n)
        return bit_map

    def _qk_train(self, q, k):
        """
        QK computation in training mode.
        """
        quant_q = self._quantize_signed_matrix_train(q, self.qk_qbit)
        quant_k = self._quantize_signed_matrix_train(k, self.qk_kbit)
        output = quant_q @ quant_k
        perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.attn_trim_noise * output
        output = output + perturb
        return output

    def _av_train(self, a, v):
        """
        AV computation in training mode.
        """
        quant_a = self._quantize_unsigned_matrix_train(a, self.av_abit)
        quant_v = self._quantize_signed_matrix_train(v, self.av_vbit)
        output = quant_a @ quant_v
        perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.attn_trim_noise * output
        output = output + perturb
        return output

    def _qk_infer(self, q, k):
        """
        QK computation in inference mode.
        """
        quant_q, scale_q = self._quantize_signed_matrix_infer(q, self.qk_qbit)
        quant_k, scale_k = self._quantize_signed_matrix_infer(k, self.qk_kbit)
        output = quant_q @ quant_k
        perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.attn_trim_noise * output
        output += perturb
        output = output * (scale_q * scale_k) / ((2.0 ** (self.qk_qbit - 1.0) - 1.0) * (2.0 ** (self.qk_kbit - 1.0) - 1.0))
        return output

    def _av_infer(self, a, v):
        """
        AV computation in inference mode.
        """
        quant_a, scale_a = self._quantize_unsigned_matrix_infer(a, self.av_abit)
        quant_v, scale_v = self._quantize_signed_matrix_infer(v, self.av_vbit)
        output = quant_a @ quant_v
        perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.attn_trim_noise * output
        output += perturb
        output = output * (scale_a * scale_v) / ((2.0 ** self.av_abit - 1.0) * (2.0 ** (self.av_vbit - 1.0) - 1.0))
        return output

    def _qk_asim(self, q, k):
        """
        QK computation in ASiM simulation.
        """
        # Quantize Q and K matrix
        quant_q, scale_q = self._quantize_signed_matrix_infer(q, self.qk_qbit)
        quant_k, scale_k = self._quantize_signed_matrix_infer(k, self.qk_kbit)
        # Decompose Q and K bit map
        q_map = self._decompose_matrix(quant_q, self.qk_qbit, signed=True).to(dtype=torch.int8)
        k_map = self._decompose_matrix(quant_k, self.qk_kbit, signed=True).to(dtype=torch.int8)
        qbit, qbatch, qhead, q_m, q_k = q_map.shape
        kbit, kbatch, khead, k_k, k_n = k_map.shape
        # Initialize output tensor
        output = torch.zeros(qbatch, qhead, q_m, k_n, device=self.device)
        # Calculate number of macro update
        macro_update = math.ceil(q_k / self.nrow)
        # Split bit map per weight update
        q_map = torch.split(q_map, self.nrow, dim=-1)
        k_map = torch.split(k_map, self.nrow, dim=-2)
        # Bit-parallel simulation
        if self.qk_bp:
            # Attention computing cycles after key bit encoding
            bp_cycle = math.ceil((self.qk_kbit - 1) / self.qk_k_enc) + 1
            # Calculate MSB encoding if there is remainder
            msb_enc = (self.qk_kbit - 1) % self.qk_k_enc
            # Loop macro update
            for n in range(macro_update):
                # Loop each query bit
                for i in range(qbit-1, -1, -1):
                    # Loop each bit-parallel key cycle
                    for j in range(bp_cycle-1, -1, -1):
                        # Initialize bit-parallel result tensor
                        bp_mac = torch.zeros_like(output, device=self.device)
                        # Bit-parallel cycle simulation
                        if j == bp_cycle - 1:
                            # Computing attention sign bit using bit-wise matmul
                            bp_mac = q_map[n][qbit-1-i].to(dtype=torch.float32) @ k_map[n][0].to(dtype=torch.float32)
                            # Scale MAC results by full range and clamp MAC to [0, 1]
                            bp_mac = torch.clamp(bp_mac / self.nrow, 0.0, 1.0)
                            # Generate rand noise from given sigma%
                            rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.qk_rand_noise_sigma / 100
                            # Generate non-linear from given sigma%
                            nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.qk_non_linear_sigma / 100
                            # Add noise and clamp real MAC to [0, 1]
                            bp_mac = bp_mac + rand_noise + nonl_noise
                            bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                            # Output MAC quantization by ADC
                            bp_mac = torch.round(bp_mac * (2.0 ** self.qk_adc_prec - 1.0))
                            # Transform ADC results to bit-serial intermediate cycle result
                            bp_mac = bp_mac / (2.0 ** self.qk_adc_prec - 1.0) * self.nrow
                            # Negative value for MSB
                            if i != qbit - 1:
                                bp_mac *= -1.0
                            # Integrate bit-shifted partial sums into output MAC
                            output += torch.mul(bp_mac, 2.0 ** (i + kbit - 1.0))
                            continue
                        if j == bp_cycle - 2:
                            # Computing branch if the MSB encoding is different from qk_k_enc
                            if msb_enc != 0:
                                # Calculate ideal bit-parallel MAC result using bit-wise matmul
                                for k in range(msb_enc-1, -1, -1):
                                    bs_mac = q_map[n][qbit-1-i].to(dtype=torch.float32) @ k_map[n][kbit-1-self.qk_k_enc*j-k].to(dtype=torch.float32)
                                    bp_mac += torch.mul(bs_mac, 2.0 ** k)
                                # Scale MAC results by full range and clamp MAC to [0, 1]
                                bp_mac = torch.clamp(bp_mac / ((2 ** msb_enc - 1) * self.nrow), 0.0, 1.0)
                                # Generate rand noise from given sigma%
                                rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.qk_rand_noise_sigma / 100
                                # Generate non-linear from given sigma%
                                nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.qk_non_linear_sigma / 100
                                # Add noise and clamp real MAC to [0, 1]
                                bp_mac = bp_mac + rand_noise + nonl_noise
                                bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                                # Output MAC quantization by ADC
                                bp_mac = torch.round(bp_mac * (2.0 ** self.qk_adc_prec - 1.0))
                                # Transform ADC results to bit-parallel intermediate cycle result
                                bp_mac = bp_mac / (2.0 ** self.qk_adc_prec - 1.0) * ((2 ** msb_enc - 1) * self.nrow)
                                # Negative value for MSB
                                if i == qbit - 1:
                                    bp_mac *= -1.0
                                # Integrate bit-shifted partial sums into output MAC
                                output += torch.mul(bp_mac, 2.0 ** (i + j * self.qk_k_enc))
                                continue
                        # Main computing branch for bit-parallel MAC result using bit-wise matmul
                        for k in range(self.qk_k_enc-1, -1, -1):
                            bs_mac = q_map[n][qbit-1-i].to(dtype=torch.float32) @ k_map[n][kbit-1-self.qk_k_enc*j-k].to(dtype=torch.float32)
                            bp_mac += torch.mul(bs_mac, 2.0 ** k)
                        # Scale MAC results by full range and clamp MAC to [0, 1]
                        bp_mac = torch.clamp(bp_mac / ((2 ** self.qk_k_enc - 1) * self.nrow), 0.0, 1.0)
                        # Generate rand noise from given sigma%
                        rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.qk_rand_noise_sigma / 100
                        # Generate non-linear from given sigma%
                        nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.qk_non_linear_sigma / 100
                        # Add noise and clamp real MAC to [0, 1]
                        bp_mac = bp_mac + rand_noise + nonl_noise
                        bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                        # Output MAC quantization by ADC
                        bp_mac = torch.round(bp_mac * (2.0 ** self.qk_adc_prec - 1.0))
                        # Transform ADC results to bit-parallel intermediate cycle result
                        bp_mac = bp_mac / (2.0 ** self.qk_adc_prec - 1.0) * ((2 ** self.qk_k_enc - 1) * self.nrow)
                        # Negative value for MSB
                        if i == qbit - 1:
                            bp_mac *= -1.0
                        # Integrate bit-shifted partial sums into output MAC
                        output += torch.mul(bp_mac, 2.0 ** (i + j * self.qk_k_enc))
            # Restore to FP32 scale
            output = output * (scale_q * scale_k) / ((2.0 ** (self.qk_qbit - 1.0) - 1.0) * (2.0 ** (self.qk_kbit - 1.0) - 1.0))
        # Bit-serial simulation
        else:
            # Loop macro update
            for n in range(macro_update):
                # Loop each query bit
                for i in range(qbit-1, -1, -1):
                    # Loop each key bit to calculate ideal bit-serial MAC result using bit-wise matmul
                    for j in range(kbit-1, -1, -1):
                        # Oversampling
                        samp_results = []
                        if i + j > qbit + kbit - self.boundary_levels - 2:
                            num_samp = self.num_samp
                        else:
                            num_samp = 1
                        for _ in range(num_samp):
                            mac_col = q_map[n][qbit-1-i].to(dtype=torch.float32) @ k_map[n][kbit-1-j].to(dtype=torch.float32)
                            # Scale MAC results by full range and clamp MAC to [0, 1]
                            mac_col = torch.clamp(mac_col / self.nrow, 0.0, 1.0)
                            # Generate rand noise from given sigma%
                            rand_noise = torch.randn(mac_col.shape, device=self.device) * self.qk_rand_noise_sigma / 100
                            # Generate non-linear from given sigma%
                            nonl_noise = torch.randn(mac_col.shape, device=self.device) / (torch.sqrt(mac_col * self.nrow + 1)) * self.qk_non_linear_sigma / 100
                            # Add noise and clamp real MAC to [0, 1]
                            mac_col = mac_col + rand_noise + nonl_noise
                            mac_col = torch.clamp(mac_col, 0.0, 1.0)
                            # Output MAC quantization by ADC
                            cur_mac_col = torch.round(mac_col * (2.0 ** self.adc_prec - 1.0))
                            samp_results.append(cur_mac_col)
                        # Majority voting to get the most common result
                        mac_col = torch.mode(torch.stack(samp_results), dim=0).values
                        # Transform ADC results to bit-serial intermediate cycle result
                        mac_col = mac_col / (2.0 ** self.qk_adc_prec - 1.0) * self.nrow
                        # Negative value for MSB
                        if (i == (qbit - 1) or j == (kbit - 1)) and ((i + j) != (qbit + kbit - 2)):
                            mac_col *= -1.0
                        # Integrate bit-shifted partial sums into output MAC
                        output += torch.mul(mac_col, 2.0 ** (i + j))
            # Restore to FP32 scale
            output = output * (scale_q * scale_k) / ((2.0 ** (self.qk_qbit - 1.0) - 1.0) * (2.0 ** (self.qk_kbit - 1.0) - 1.0))

        return output

    def _av_asim(self, a, v):
        """
        AV computation in ASiM simulation.
        """
        # Quantize Score and V matrix
        quant_a, scale_a = self._quantize_unsigned_matrix_infer(a, self.av_abit)
        quant_v, scale_v = self._quantize_signed_matrix_infer(v, self.av_vbit)
        # Decompose Score and V bit map
        score_map = self._decompose_matrix(quant_a, self.av_abit, signed=False).to(dtype=torch.int8)
        v_map = self._decompose_matrix(quant_v, self.av_vbit, signed=True).to(dtype=torch.int8)
        abit, abatch, ahead, a_m, a_k = score_map.shape
        vbit, vbatch, vhead, v_k, v_n = v_map.shape
        # Initialize output tensor
        output = torch.zeros(abatch, ahead, a_m, v_n, device=self.device)
        # Calculate number of macro update
        macro_update = math.ceil(a_k / self.nrow)
        # Split bit map per weight update
        score_map = torch.split(score_map, self.nrow, dim=-1)
        v_map = torch.split(v_map, self.nrow, dim=-2)
        # Bit-parallel simulation
        if self.av_bp:
            # Attention computing cycles after score bit encoding
            bp_cycle = math.ceil(self.av_abit / self.av_a_enc)
            # Calculate MSB encoding if there is remainder
            msb_enc = self.av_abit % self.av_a_enc
            # Loop macro update
            for n in range(macro_update):
                # Loop each value bit
                for i in range(vbit-1, -1, -1):
                    # Loop each bit-parallel attention score cycle
                    for j in range(bp_cycle-1, -1, -1):
                        # Initialize bit-parallel result tensor
                        bp_mac = torch.zeros_like(output, device=self.device)
                        # Bit-parallel cycle simulation
                        if j == bp_cycle - 1:
                            # Computing branch if the MSB encoding is different from enc_bit
                            if msb_enc != 0:
                                # Calculate ideal bit-parallel MAC result using bit-wise matmul
                                for k in range(msb_enc-1, -1, -1):
                                    bs_mac = score_map[n][abit-1-self.av_a_enc*j-k].to(dtype=torch.float32) @ v_map[n][vbit-1-i].to(dtype=torch.float32)
                                    bp_mac += torch.mul(bs_mac, 2.0 ** k)
                                # Scale MAC results by full range and clamp MAC to [0, 1]
                                bp_mac = torch.clamp(bp_mac / ((2 ** msb_enc - 1) * self.nrow), 0.0, 1.0)
                                # Generate rand noise from given sigma%
                                rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.av_rand_noise_sigma / 100
                                # Generate non-linear from given sigma%
                                nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.av_non_linear_sigma / 100
                                # Add noise and clamp real MAC to [0, 1]
                                bp_mac = bp_mac + rand_noise + nonl_noise
                                bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                                # Output MAC quantization by ADC
                                bp_mac = torch.round(bp_mac * (2.0 ** self.av_adc_prec - 1.0))
                                # Transform ADC results to bit-parallel intermediate cycle result
                                bp_mac = bp_mac / (2.0 ** self.av_adc_prec - 1.0) * ((2 ** msb_enc - 1) * self.nrow)
                                # Negative value for MSB
                                if i == vbit - 1:
                                    bp_mac *= -1.0
                                # Integrate bit-shifted partial sums into output activations
                                output += torch.mul(bp_mac, 2.0 ** (i + j * self.av_a_enc))
                                continue
                        # Main computing branch for bit-parallel MAC result using bit-wise matmul
                        for k in range(self.av_a_enc-1, -1, -1):
                            bs_mac = score_map[n][abit-1-self.av_a_enc*j-k].to(dtype=torch.float32) @ v_map[n][vbit-1-i].to(dtype=torch.float32)
                            bp_mac += torch.mul(bs_mac, 2.0 ** k)
                        # Scale MAC results by full range and clamp MAC to [0, 1]
                        bp_mac = torch.clamp(bp_mac / ((2 ** self.av_a_enc - 1) * self.nrow), 0.0, 1.0)
                        # Generate rand noise from given sigma%
                        rand_noise = torch.randn(bp_mac.shape, device=self.device) * self.av_rand_noise_sigma / 100
                        # Generate non-linear from given sigma%
                        nonl_noise = torch.randn(bp_mac.shape, device=self.device) / (torch.sqrt(bp_mac * self.nrow + 1)) * self.av_non_linear_sigma / 100
                        # Add noise and clamp real MAC to [0, 1]
                        bp_mac = bp_mac + rand_noise + nonl_noise
                        bp_mac = torch.clamp(bp_mac, 0.0, 1.0)
                        # Output MAC quantization by ADC
                        bp_mac = torch.round(bp_mac * (2.0 ** self.av_adc_prec - 1.0))
                        # Transform ADC results to bit-parallel intermediate cycle result
                        bp_mac = bp_mac / (2.0 ** self.av_adc_prec - 1.0) * ((2 ** self.av_a_enc - 1) * self.nrow)
                        # Negative value for MSB
                        if i == vbit - 1:
                            bp_mac *= -1.0
                        # Integrate bit-shifted partial sums into output MAC
                        output += torch.mul(bp_mac, 2.0 ** (i + j * self.av_a_enc))
            # Restore to FP32 scale
            output = output * (scale_a * scale_v) / ((2.0 ** (self.av_vbit - 1.0) - 1.0) * (2.0 ** self.av_abit - 1.0))
        # Bit-serial simulation
        else:
            # Loop macro update
            for n in range(macro_update):
                # Loop each value bit
                for i in range(vbit-1, -1, -1):
                    # Loop each attention score bit to calculate ideal bit-serial MAC result using bit-wise matmul
                    for j in range(abit-1, -1, -1):
                        # Oversampling
                        samp_results = []
                        if i + j > vbit + abit - self.boundary_levels - 2:
                            num_samp = self.num_samp
                        else:
                            num_samp = 1
                        for _ in range(num_samp):
                            mac_col = score_map[n][abit-1-j].to(dtype=torch.float32) @ v_map[n][vbit-1-i].to(dtype=torch.float32)
                            # Scale MAC results by full range and clamp MAC to [0, 1]
                            mac_col = torch.clamp(mac_col / self.nrow, 0.0, 1.0)
                            # Generate rand noise from given sigma%
                            rand_noise = torch.randn(mac_col.shape, device=self.device) * self.av_rand_noise_sigma / 100
                            # Generate non-linear from given sigma%
                            nonl_noise = torch.randn(mac_col.shape, device=self.device) / (torch.sqrt(mac_col * self.nrow + 1)) * self.av_non_linear_sigma / 100
                            # Add noise and clamp real MAC to [0, 1]
                            mac_col = mac_col + rand_noise + nonl_noise
                            mac_col = torch.clamp(mac_col, 0.0, 1.0)
                            # Output MAC quantization by ADC
                            cur_mac_col = torch.round(mac_col * (2.0 ** self.adc_prec - 1.0))
                            samp_results.append(cur_mac_col)
                        # Majority voting to get the most common result
                        mac_col = torch.mode(torch.stack(samp_results), dim=0).values
                        # Transform ADC results to bit-serial intermediate cycle result
                        mac_col = mac_col / (2.0 ** self.av_adc_prec - 1.0) * self.nrow
                        # Negative value for MSB
                        if i == vbit - 1:
                            mac_col *= -1.0
                        # Integrate bit-shifted partial sums into output MAC
                        output += torch.mul(mac_col, 2.0 ** (i + j))
            # Restore to FP32 scale
            output = output * (scale_a * scale_v) / ((2.0 ** (self.av_vbit - 1.0) - 1.0) * (2.0 ** self.av_abit - 1.0))

        return output

    def forward(self, x):
        """
        Forward call of ASiMMultiheadAttention with selective operation mode.
        input: input activation tensor
        return: output activation tensor
        """
        if self.mode == 'Train':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = self._qk_train(q, k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout_layer(attn)

            x = self._av_train(attn, v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)

        if self.mode == 'Inference':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = self._qk_infer(q, k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout_layer(attn)

            x = self._av_infer(attn, v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)

        if self.mode == 'Simulation':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = self._qk_asim(q, k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout_layer(attn)

            x = self._av_asim(attn, v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)

        return x


if __name__ == '__main__':  # testbench
    pass
