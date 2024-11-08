"""

VGG models for the simulation framework.

"""

import torch
import torch.nn as nn
from main.config import cfg
from module.basic_module import QuantConv2d, PACT
from module.asim_conv import ASiMConv2d
from module.asim_linear import ASiMLinear


def asim_conv(in_planes, out_planes, stride=1):
    return ASiMConv2d(in_planes,
                      out_planes,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=True,
                      wbit=cfg.asim_conv_wbit,
                      xbit=cfg.asim_conv_xbit,
                      adc_prec=cfg.asim_conv_adc_prec,
                      nrow=cfg.asim_conv_nrow,
                      rand_noise_sigma=cfg.asim_linear_rand_noise_sigma,
                      non_linear_sigma=cfg.asim_linear_non_linear_sigma,
                      act_enc=cfg.asim_act_enc,
                      signed_act=cfg.asim_signed_act,
                      hybrid_levels=cfg.asim_cnn_hybrid_levels,
                      mode=cfg.asim_conv_mode,
                      trim_noise=cfg.asim_conv_trim_noise,
                      device=cfg.device)


def asim_affine(in_features, out_features):
    return ASiMLinear(in_features,
                      out_features,
                      bias=True,
                      wbit=cfg.asim_linear_wbit,
                      xbit=cfg.asim_linear_xbit,
                      adc_prec=cfg.asim_linear_adc_prec,
                      nrow=cfg.asim_linear_nrow,
                      rand_noise_sigma=cfg.asim_linear_rand_noise_sigma,
                      non_linear_sigma=cfg.asim_linear_non_linear_sigma,
                      act_enc=cfg.asim_linear_act_enc,
                      signed_act=cfg.asim_linear_signed_act,
                      layer=cfg.asim_linear_layer,
                      hybrid_levels=cfg.asim_cnn_hybrid_levels,
                      mode=cfg.asim_linear_mode,
                      trim_noise=cfg.asim_linear_trim_noise,
                      device=cfg.device)


def quant_conv(in_planes, out_planes, stride=1):
    return QuantConv2d(in_planes,
                       out_planes,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=True,
                       wbit=cfg.quant_conv_wbit,
                       xbit=cfg.quant_conv_xbit,
                       signed_act=True,
                       mode=cfg.quant_conv_mode,
                       device=cfg.device)


if cfg.PACT:
    relu = PACT()
else:
    relu = nn.ReLU(inplace=True)


class VGG(nn.Module):
    def __init__(self, features, num_classes=cfg.cls_num, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            asim_affine(512 * 7 * 7, 4096),
            relu,
            nn.Dropout(),
            asim_affine(4096, 4096),
            relu,
            nn.Dropout(),
            asim_affine(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ASiMConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, ASiMLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(str, batch_norm=False):
    layers = []
    if batch_norm:
        layers += [quant_conv(3, 64),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True)]
    else:
        layers += [quant_conv(3, 64),
                   nn.ReLU(inplace=True)]

    in_channels = 64
    for v in str:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = asim_conv(in_channels, v)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), relu]
            else:
                layers += [conv2d, relu]
            in_channels = v
    return nn.Sequential(*layers)


if cfg.large_model:

    str = {
        "A": ["M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "B": [64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "D": [64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "E": [64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    }
else:
    str = {
        "A": [128, "M", 256, 256, 512, 512, 512, 512, "M"],
        "B": [64, 128, 128, "M", 256, 256, 512, 512, 512, 512, "M"],
        "D": [64, 128, 128, "M", 256, 256, 256, 512, 512, 512, 512, 512, 512, "M"],
        "E": [64, 128, 128, "M", 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, "M"]
    }


def vgg11_asim(**kwargs):
    model = VGG(make_layers(str['A']), **kwargs)
    return model


def vgg11_bn_asim(**kwargs):
    model = VGG(make_layers(str['A'], batch_norm=True), **kwargs)
    return model


def vgg13_asim(**kwargs):
    model = VGG(make_layers(str['B']), **kwargs)
    return model


def vgg13_bn_asim(**kwargs):
    model = VGG(make_layers(str['B'], batch_norm=True), **kwargs)
    return model


def vgg16_asim(**kwargs):
    model = VGG(make_layers(str['D']), **kwargs)
    return model


def vgg16_bn_asim(**kwargs):
    model = VGG(make_layers(str['D'], batch_norm=True), **kwargs)
    return model


def vgg19_asim(**kwargs):
    model = VGG(make_layers(str['E']), **kwargs)
    return model


def vgg19_bn_asim(**kwargs):
    model = VGG(make_layers(str['E'], batch_norm=True), **kwargs)
    return model


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = vgg16_bn_asim()
    model.to(device=device)

    for name, module in model.named_modules():
        print('Layer name: {}, Layer instance: {}'.format(name, module))

    # Forward
    fake_img = torch.randn((1, 3, 28, 28), device=device)
    output = model(fake_img)
    print(output.shape)
