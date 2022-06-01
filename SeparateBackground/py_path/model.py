# -*- coding : utf-8 -*-
# @Time      : 2022/4/21 19:51
# @Author    : Zong
# @File      : model.py
# @Software  : PyCharm
# @Function  : 模型设置文件
# @ChangeLog :
import torch
import torchvision.models as models
from torch import nn
import sys
sys.path.append("..")
import config


# 修改网络结构
MODEL = models.segmentation.deeplabv3_resnet50(pretrained = True)

# 第一层输入
# deeplabv3_resnet50.backbone.conv1 = nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)

MODEL.classifier[1] = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
MODEL.classifier[2] = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# 最后一层输出
MODEL.classifier[4] = nn.Conv2d(128, config.CLASSES_NUM_EXCEPT_BG, kernel_size=(1, 1), stride=(1, 1))

# 模型放入运行环境
MODEL = MODEL.to(config.DEVICE)


# # 测试
print(MODEL)
# input = torch.empty(64, 3, 512, 512)
# input = input.view(-1, 1, 1, 3)
# deeplabv3_resnet50(input)


# cuda
# print(torch.version.cuda)