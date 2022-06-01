# -*- coding : utf-8 -*-
# @Time      : 2022/4/21 22:33
# @Author    : Zong
# @File      : new_test.py
# @Software  : PyCharm
# @Function  : 模型测试
# @ChangeLog :


import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import argmax
from torchvision import transforms
from torch.autograd import Variable
from py_path.dataset import image_preprocess
import config


# 加载模型，打开验证模式
model = torch.load("./model_path/model_5.pth")
model.eval()

# 加载图片
file_name_path = "./dataset_path/train_datasets_document_detection_0411/images/"
filename1 = "image_02000.jpg"
filename2 = "image_02001.jpg"
filename3 = "image_02002.jpg"

filenames = [filename1, filename2, filename3]
titles = []
images = []

# 遍历
for single_filename in filenames:
    img = cv2.imread(file_name_path + single_filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (config.WIDTH, config.HEIGHT), interpolation = cv2.INTER_AREA)
    img_rgb = img_rgb / 255

    # 输入模型
    input_tensor = image_preprocess(img_rgb)
    input_batch = Variable(torch.unsqueeze(input_tensor, dim=0).float(), requires_grad=False)

    # 获取输出
    output = model(input_batch)["out"][0]
    output_predictions = output.argmax(0)
    img = output_predictions * 255
    imgArray = np.array(img)

    # 图像转黑白
    height, width = imgArray.shape
    img3Array = np.ones([height, width, 3], dtype='uint8')
    img3Array[:, :, 0] = imgArray[:, :]
    img3Array[:, :, 1] = imgArray[:, :]
    img3Array[:, :, 2] = imgArray[:, :]

    # 显示
    img_name = single_filename.split(".")[0]
    titles.append(img_name)
    titles.append(img_name + "_mask")
    images.append(img_rgb)
    images.append(img3Array)


# 显示图形
for i in range(len(images)):
    row = len(filenames)
    plt.subplot(row, int((len(images) + 1) / row), i + 1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

