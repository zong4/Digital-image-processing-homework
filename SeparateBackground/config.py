# -*- coding : utf-8 -*-
# @Time      : 2022/4/23 20:01
# @Author    : Zong
# @File      : config.py
# @Software  : PyCharm
# @Function  : 全局设置
# @ChangeLog :


import torch
from torch import nn


# 运行环境
DEVICE = torch.device("cpu")


# 模型设置
CLASSES_NUM_EXCEPT_BG = 2
MODEL_STORE_DIR = "./model_path/"


# 数据集设置
# 缩放图像大小，该数据集图像大小为512
HEIGHT = 512
WIDTH = 512

# 数据集大小
IMG_ALL_NUM = 2797 -1 + 1
TRAIN_IMAGE_NUM = 1000  # 1 ~ 2797
TEST_IMAGE_NUM = 200  # 倒数测试图片
REVERSE_START_INDEX_FROM_ZERO = IMG_ALL_NUM - TEST_IMAGE_NUM


# Dataloader
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

SHUFFLE = True  # 是否刷新dataloader


# 损失函数
LOSS_FUN = nn.CrossEntropyLoss().to(DEVICE)


# 优化器，可选SGD/
OPTIM = "SGD"


# 学习率
LEARN_RATE = 1e-3


# 模型训练设置
# 轮数
EPOCH = 10

# print间隔
PRINT_INTERVAL = 4
