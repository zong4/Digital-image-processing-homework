# -*- coding : utf-8 -*-
# @Time      : 2022/4/21 18:54
# @Author    : Zong
# @File      : dataset.py
# @Software  : PyCharm
# @Function  : 数据集
# @ChangeLog :


import cv2
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
# from py_path import norm
sys.path.append("..")
import config


# 图片预处理
image_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.536022248710712, 0.5205804213979173, 0.5056642637564716], std=[0.1966466496167661, 0.20699665603536513, 0.22172627152124336]),
        ])

mask_preprocess = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5692316774854472, 0.5692316774854472, 0.5692316774854472], std=[0.450555273117079, 0.450555273117079, 0.450555273117079]),
        ])


# 继承Dataset
class train_dataset(data.Dataset):
    # 初始化
    def __init__(self, root):
        super(train_dataset, self).__init__()
        self.root = root

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        # 数据集从1开始
        item = item + 1

        # 图片索引号
        item = "%05d" % item

        # 主图
        img = cv2.imread(self.root + '\\images\\image_{}.jpg'.format(item))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (config.WIDTH, config.HEIGHT), interpolation = cv2.INTER_AREA)

        # 归一化，标准化
        img_rgb = img_rgb / 255.0
        img_tensor = image_preprocess(img_rgb)

        # 蒙板
        mask = cv2.imread(self.root + '\\segments\\image_{}.png'.format(item))
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(mask_rgb, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)

        # 归一化，标准化
        mask_rgb = mask_rgb / 255.0
        mask_tensor = mask_preprocess(mask_rgb)

        # 去掉一维
        mask_tensor = mask_tensor[:][0][:][:]

        return img_tensor, mask_tensor

    # 获取数据集样本个数
    def __len__(self):
        return config.TRAIN_IMAGE_NUM


class test_dataset(data.Dataset):
    # 初始化
    def __init__(self, root):
        super(test_dataset, self).__init__()
        self.root = root

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        # 最后一张为IMG_ALL_NUM
        item = config.REVERSE_START_INDEX_FROM_ZERO + item + 1

        # 图片索引号
        item = "%05d" % item

        # 主图
        img = cv2.imread(self.root + '\\images\\image_{}.jpg'.format(item))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 归一化，标准化
        img_rgb = img_rgb / 255
        img_tensor = image_preprocess(img_rgb)

        # 蒙板
        mask = cv2.imread(self.root + '\\segments\\image_{}.png'.format(item))
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # 归一化，标准化
        mask_rgb = mask_rgb / 255
        mask_tensor = mask_preprocess(mask_rgb)

        # 去掉一维
        mask_tensor = mask_tensor[:][0][:][:]

        return img_tensor, mask_tensor

    # 获取数据集样本个数
    def __len__(self):
        return config.TEST_IMAGE_NUM


# 数据集实例化
segdataset_train_instan = train_dataset("dataset_path\\train_datasets_document_detection_0411")
segdataset_test_instan = test_dataset("dataset_path\\train_datasets_document_detection_0411")
# a, b = test[1]
#
# print(a)


# 加载DataLoader
train_dataloader = DataLoader(segdataset_train_instan, batch_size = config.TRAIN_BATCH_SIZE, shuffle = config.SHUFFLE)
test_dataloader = DataLoader(segdataset_test_instan, batch_size = config.TEST_BATCH_SIZE, shuffle = config.SHUFFLE)


# loss = nn.CrossEntropyLoss()
# a = torch.empty(1,1, 3, 3)
# print(a.shape)
# b = torch.empty(1,1, 3, 3)
# result = loss(a,b)
# print("%05d" % 1)

