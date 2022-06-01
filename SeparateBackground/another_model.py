# -*- coding : utf-8 -*-
# @Time      : 2022/5/4 22:30
# @Author    : Zong
# @File      : another_model.py
# @Software  : PyCharm
# @Function  : 
# @ChangeLog :


import torch
from torch import nn
# from torchvision.models._utils import IntermediateLayerGetter
import config


# print(IntermediateLayerGetter)


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels = 64, middle_channels = 64,out_channels = 256, is_downsample = 0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        if is_downsample:
            self.downsample = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
              nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )



class IntermediateLayerGetter(nn.Sequential):
    def __init__(self, in_channels = 3, middle_channels = 64):
        super(IntermediateLayerGetter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # layer1
        in_channels = middle_channels  # 64
        middle_channels = middle_channels  # 64
        out_channels = 4 * middle_channels  # 256
        self.layer1 = nn.Sequential(
            Bottleneck(in_channels, middle_channels, out_channels, is_downsample = 1),
            Bottleneck(out_channels, middle_channels, out_channels),
            Bottleneck(out_channels, middle_channels, out_channels),
        )

        # layer2
        in_channels  = out_channels  # 256
        middle_channels = 2 * middle_channels  # 128
        out_channels = 4 * middle_channels  # 512
        self.layer2 = nn.Sequential(
            Bottleneck(in_channels, middle_channels, out_channels, is_downsample = 1),
            Bottleneck(out_channels, middle_channels, out_channels),
            Bottleneck(out_channels, middle_channels, out_channels),
            Bottleneck(out_channels, middle_channels, out_channels),
        )

        # layer3
        in_channels = out_channels  # 512
        middle_channels = 2 * middle_channels  # 256
        out_channels = 4 * middle_channels  # 1024
        self.layer3 = nn.Sequential(
            Bottleneck(in_channels, middle_channels, out_channels, is_downsample=1),
            Bottleneck(out_channels, middle_channels, out_channels),
            Bottleneck(out_channels, middle_channels, out_channels),
            Bottleneck(out_channels, middle_channels, out_channels),
        )

        # layer4
        in_channels = out_channels  # 1024
        middle_channels = 2 * middle_channels  # 512
        out_channels = 4 * middle_channels  # 2048
        self.layer4 = nn.Sequential(
            Bottleneck(in_channels, middle_channels, out_channels, is_downsample=1),
            Bottleneck(out_channels, middle_channels, out_channels),
            Bottleneck(out_channels, middle_channels, out_channels),
        )



# ASPPConv
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# ASPPPooling
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

    def forward(self, input):
        size = input.shape[-2 : ]
        for single_model in self:
            input = single_model(input)
        return nn.functional.interpolate(input, size=size, mode='bilinear', align_corners=False)

# ASPP
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            )
        )

        for i in range(0, times):
            modules.append(ASPPConv(in_channels, out_channels))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
        )

    def forward(self, input):
        result = []
        for single_conv in self.convs:
            result.append(single_conv(input))

        # 砍掉多余维度
        result = torch.cat(result, dim=1)
        return self.project(result)

# 主干模型
class DeepLabHead(nn.ModuleList):
    def __init__(self, in_channels = 2048, out_channels = 256, times = 3, middle_channels = 128, num_classes = 2):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, out_channels, times),
            nn.Conv2d(out_channels, middle_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(middle_channels, num_classes, kernel_size=(1, 1), stride=(1, 1)),
        )
        # self.classifier = nn.ModuleList(modules)

    def forward(self, input):
        result = []
        for single_model in self.classifier:
            result.append(single_model(input))

        # 砍掉多余维度
        result = torch.cat(result, dim=1)
        return self.project(result)


class DeepLabV3(nn.Module):
    def __init__(self, in_channels = 3, middle_channels = 64, max_channels = 2048):
        super(DeepLabHead, self).__init__()
        self.backbone = IntermediateLayerGetter(in_channels, middle_channels)
        self.classifier = DeepLabHead(in_channels, max_channels, max_channels)



MODEL = DeepLabV3(in_channels = 2048, out_channels = 256, times = 3, middle_channels = 128, num_classes = 2)
MODEL = MODEL.to(config.DEVICE)


print(MODEL)