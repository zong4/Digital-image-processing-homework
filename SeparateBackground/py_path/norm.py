# -*- coding : utf-8 -*-
# @Time      : 2022/4/24 13:02
# @Author    : Zong
# @File      : normalization.py
# @Software  : PyCharm
# @Function  : 图像标准化
# @ChangeLog :


import os
import cv2
import numpy as np
import sys
sys.path.append("..")
import config


# 读取数据
images_path = "../dataset_path/train_datasets_document_detection_0411/images"
masks_path = "../dataset_path/train_datasets_document_detection_0411/segments"
images = os.listdir(images_path)
masks = os.listdir(masks_path)

# 存放数据
per_image_R_mean = []
per_image_G_mean = []
per_image_B_mean = []
per_image_R_std = []
per_image_G_std = []
per_image_B_std = []

per_mask_mean = []
per_mask_std = []

# 训练集
for index in range(config.TRAIN_IMAGE_NUM):
    # 读取图片
    img = cv2.imread(images_path + '/' + images[index])
    mask = cv2.imread(masks_path + '/' + masks[index], 0)  # 黑白读取

    # 保持相同处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    img_rgb = img_rgb / 255.0

    mask = cv2.resize(mask, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    mask = mask / 255.0

    # 计算mean和std
    mean, std = cv2.meanStdDev(img_rgb)
    per_image_R_mean.append(mean[0])
    per_image_G_mean.append(mean[1])
    per_image_B_mean.append(mean[2])
    per_image_R_std.append(std[0])
    per_image_G_std.append(std[1])
    per_image_B_std.append(std[2])

    mean, std = cv2.meanStdDev(mask)
    per_mask_mean.append(mean[0])
    per_mask_std.append(std[0])

# 测试集
for index in range(config.TEST_IMAGE_NUM):
    # 读取图片
    img = cv2.imread(images_path + '/' + images[config.REVERSE_START_INDEX_FROM_ZERO + index])
    mask = cv2.imread(masks_path + '/' + masks[config.REVERSE_START_INDEX_FROM_ZERO + index], 0)  # 黑白读取

    # 保持相同处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    img_rgb = img_rgb / 255.0

    mask = cv2.resize(mask, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    mask = mask / 255.0

    # 计算mean和std
    mean, std = cv2.meanStdDev(img_rgb)
    per_image_R_mean.append(mean[0])
    per_image_G_mean.append(mean[1])
    per_image_B_mean.append(mean[2])
    per_image_R_std.append(std[0])
    per_image_G_std.append(std[1])
    per_image_B_std.append(std[2])

    mean, std = cv2.meanStdDev(mask)
    per_mask_mean.append(mean[0])
    per_mask_std.append(std[0])


# 计算结果
# print(len(per_mask_std))
images_R_mean = np.mean(per_image_R_mean)
images_G_mean = np.mean(per_image_G_mean)
images_B_mean = np.mean(per_image_B_mean)
images_R_std = np.mean(per_image_R_std)
images_G_std = np.mean(per_image_G_std)
images_B_std = np.mean(per_image_B_std)

masks_mean = np.mean(per_mask_mean)
masks_std = np.mean(per_mask_std)

# 打印
print(images_R_mean, images_G_mean, images_B_mean)
print(images_R_std, images_G_std, images_B_std)
print(masks_mean, masks_std)

