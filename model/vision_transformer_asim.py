"""

Vision Transformer models for the simulation framework.

"""

import torch
import torch.nn as nn
from main.config import cfg
from module.basic_module import QuantConv2d
from module.asim_linear import ASiMLinear
from module.asim_multiheadattention import ASiMMultiheadAttention


def attn(embed_dim, num_heads, dropout=0.1):
    return ASiMMultiheadAttention(embed_dim,
                                  num_heads,
                                  dropout=dropout,
                                  qk_qbit=cfg.asim_vit_attn_qk_qbit,
                                  qk_kbit=cfg.asim_vit_attn_qk_kbit,
                                  av_abit=cfg.asim_vit_attn_av_abit,
                                  av_vbit=cfg.asim_vit_attn_av_vbit,
                                  proj_wbit=cfg.asim_vit_attn_proj_wbit,
                                  proj_xbit=cfg.asim_vit_attn_proj_xbit,
                                  qk_adc_prec=cfg.asim_vit_attn_qk_adc_prec,
                                  av_adc_prec=cfg.asim_vit_attn_av_adc_prec,
                                  proj_adc_prec=cfg.asim_vit_attn_proj_adc_prec,
                                  nrow=cfg.asim_vit_attn_nrow,
                                  qk_rand_noise_sigma=cfg.asim_vit_attn_qk_rand_noise_sigma,
                                  av_rand_noise_sigma=cfg.asim_vit_attn_av_rand_noise_sigma,
                                  proj_rand_noise_sigma=cfg.asim_vit_attn_proj_rand_noise_sigma,
                                  qk_non_linear_sigma=cfg.asim_vit_attn_qk_non_linear_sigma,
                                  av_non_linear_sigma=cfg.asim_vit_attn_av_non_linear_sigma,
                                  proj_non_linear_sigma=cfg.asim_vit_attn_proj_non_linear_sigma,
                                  qk_k_enc=cfg.asim_vit_attn_qk_k_enc,
                                  av_a_enc=cfg.asim_vit_attn_av_a_enc,
                                  proj_act_enc=cfg.asim_vit_attn_proj_act_enc,
                                  hybrid_levels=cfg.asim_vit_hybrid_levels,
                                  mode=cfg.asim_vit_attn_mode,
                                  attn_trim_noise=cfg.asim_vit_attn_attn_trim_noise,
                                  proj_trim_noise=cfg.asim_vit_attn_proj_trim_noise,
                                  device=cfg.device)


def mlp(in_features, out_features):
    return ASiMLinear(in_features,
                      out_features,
                      bias=True,
                      wbit=cfg.asim_vit_mlp_wbit,
                      xbit=cfg.asim_vit_mlp_xbit,
                      adc_prec=cfg.asim_vit_mlp_adc_prec,
                      nrow=cfg.asim_vit_mlp_nrow,
                      rand_noise_sigma=cfg.asim_vit_mlp_rand_noise_sigma,
                      non_linear_sigma=cfg.asim_vit_mlp_non_linear_sigma,
                      act_enc=cfg.asim_vit_mlp_act_enc,
                      signed_act=True,
                      layer='proj',
                      hybrid_levels=cfg.asim_vit_hybrid_levels,
                      mode=cfg.asim_vit_mlp_mode,
                      trim_noise=cfg.asim_vit_mlp_trim_noise,
                      device=cfg.device)


def fc(in_features, out_features):
    return ASiMLinear(in_features,
                      out_features,
                      bias=True,
                      wbit=cfg.asim_vit_fc_wbit,
                      xbit=cfg.asim_vit_fc_xbit,
                      adc_prec=cfg.asim_vit_fc_adc_prec,
                      nrow=cfg.asim_vit_fc_nrow,
                      rand_noise_sigma=cfg.asim_vit_fc_rand_noise_sigma,
                      non_linear_sigma=cfg.asim_vit_fc_non_linear_sigma,
                      act_enc=cfg.asim_vit_fc_act_enc,
                      signed_act=True,
                      layer='fc',
                      hybrid_levels=cfg.asim_vit_hybrid_levels,
                      mode=cfg.asim_vit_fc_mode,
                      trim_noise=cfg.asim_vit_fc_trim_noise,
                      device=cfg.device)


def quant_conv(in_planes, out_planes, kernel_size, stride):
    return QuantConv2d(in_planes,
                       out_planes,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=0,
                       bias=True,
                       wbit=cfg.asim_vit_quant_conv_wbit,
                       xbit=cfg.asim_vit_quant_conv_xbit,
                       signed_act=True,
                       mode=cfg.asim_vit_quant_conv_mode,
                       device=cfg.device)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        if cfg.vit_quant_conv_proj:
            self.conv_proj = quant_conv(in_channels, out_channels, patch_size, patch_size)
        else:
            self.conv_proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        return x


class MLPBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim=None, out_dim=None, dropout=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        mlp_dim = mlp_dim or in_dim
        self.mlp1 = mlp(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.mlp2 = mlp(mlp_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.mlp2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout):
        super().__init__()

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = attn(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(in_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, input):
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)

        return x + y


class Encoder(nn.Module):
    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, input):
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim,
                 dropout, attention_dropout, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embedding = PatchEmbedding(image_size, patch_size, 3, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length = self.patch_embedding.num_patches + 1
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout)
        self.heads = fc(hidden_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.heads(x)

        return x


def vit_b_16_asim():
    model = VisionTransformer(image_size=cfg.image_size,
                              patch_size=16,
                              num_layers=12,
                              num_heads=12,
                              hidden_dim=768,
                              mlp_dim=3072,
                              dropout=0.1,
                              attention_dropout=0.1,
                              num_classes=cfg.cls_num)
    return model


def vit_b_32_asim():
    model = VisionTransformer(image_size=cfg.image_size,
                              patch_size=32,
                              num_layers=12,
                              num_heads=12,
                              hidden_dim=768,
                              mlp_dim=3072,
                              dropout=0.1,
                              attention_dropout=0.1,
                              num_classes=cfg.cls_num)
    return model


def vit_l_16_asim():
    model = VisionTransformer(image_size=cfg.image_size,
                              patch_size=16,
                              num_layers=24,
                              num_heads=16,
                              hidden_dim=1024,
                              mlp_dim=4096,
                              dropout=0.1,
                              attention_dropout=0.1,
                              num_classes=cfg.cls_num)
    return model


def vit_l_32_asim():
    model = VisionTransformer(image_size=cfg.image_size,
                              patch_size=32,
                              num_layers=24,
                              num_heads=16,
                              hidden_dim=1024,
                              mlp_dim=4096,
                              dropout=0.1,
                              attention_dropout=0.1,
                              num_classes=cfg.cls_num)
    return model


def vit_h_14_asim():
    model = VisionTransformer(image_size=cfg.image_size,
                              patch_size=14,
                              num_layers=32,
                              num_heads=16,
                              hidden_dim=1280,
                              mlp_dim=5120,
                              dropout=0.1,
                              attention_dropout=0.1,
                              num_classes=cfg.cls_num)
    return model
