# -*- coding : utf-8 -*-
# @Time      : 2022/4/14 17:26
# @Author    : Zong
# @File      : another.py
# @Software  : PyCharm
# @Function  : 
# @ChangeLog :


import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt


# 读取图片
img = cv2.imread('head128.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = img_rgb.shape
# print(height)


# 空间均值滤波
def space_average_blur(img_input, kernels_size):
    # 定义卷积核
    kernels = np.ones([kernels_size, kernels_size], dtype='uint8')
    padding_size = int((kernels_size - 1) / 2)

    # 定义扩展图形, 必须整数矩阵
    img_after_process = np.zeros([height, width, channels], dtype='uint8')
    img_extension = np.zeros([height + 2 * padding_size, width + 2 * padding_size, channels], dtype='uint8')
    img_extension[padding_size : padding_size + height, padding_size : padding_size + width, :] = img_input[0 : height, 0 : width, :]

    # 均值滤波
    for row in range(0, height):
        for col in range(0, width):
            # RGB三色域
            img_after_process[row, col, 0] = np.sum(np.multiply(img_extension[row : row + kernels_size, col : col + kernels_size, 0], kernels)) / kernels_size / kernels_size
            img_after_process[row, col, 1] = np.sum(
                np.multiply(img_extension[row: row + kernels_size, col : col + kernels_size, 1], kernels)) / kernels_size / kernels_size
            img_after_process[row, col, 2] = np.sum(
                np.multiply(img_extension[row: row + kernels_size, col : col + kernels_size, 2], kernels)) / kernels_size / kernels_size

    return img_after_process


# 空间锐化
def space_sharpen(img_input):
    # 定义卷积核
    kernels_size = 3
    kernels = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype='uint8')
    padding_size = int((kernels_size - 1) / 2)

    # 定义扩展图形, 必须整数矩阵
    img_after_process = np.zeros([height, width, channels], dtype='uint8')
    img_extension = np.zeros([height + 2 * padding_size, width + 2 * padding_size, channels], dtype='uint8')
    img_extension[padding_size : padding_size + height, padding_size : padding_size + width, :] = img_input[0 : height, 0 : width, :]

    for row in range(0, height):
        for col in range(0, width):
            # RGB三色域
            img_after_process[row, col, 0] = np.sum(np.multiply(img_extension[row : row + kernels_size, col : col + kernels_size, 0], kernels)) / kernels_size / kernels_size
            img_after_process[row, col, 1] = np.sum(
                np.multiply(img_extension[row: row + kernels_size, col : col + kernels_size, 1], kernels)) / kernels_size / kernels_size
            img_after_process[row, col, 2] = np.sum(
                np.multiply(img_extension[row: row + kernels_size, col : col + kernels_size, 2], kernels)) / kernels_size / kernels_size

    return img_after_process


# 频域变换
def frequency_change(img_input, radius, method):
    # 傅里叶正变换
    fouier_r = np.fft.fft2(img_input[:, :, 0])
    fouier_g = np.fft.fft2(img_input[:, :, 1])
    fouier_b = np.fft.fft2(img_input[:, :, 2])
    fouier_center_r = np.fft.fftshift(fouier_r)
    fouier_center_g = np.fft.fftshift(fouier_g)
    fouier_center_b = np.fft.fftshift(fouier_b)

    # 滤波掩膜
    mask = np.zeros((height, width), dtype='uint8')
    # 将距离频谱中心距离小于D的低通信息部分设置为1
    for row in range(height):
        for col in range(width):
            if(np.sqrt(abs(height / 2 - row) * abs(height / 2 - row) + abs(width / 2 - col) * abs(width / 2 - col)) <= radius * radius):
                mask[row, col] = 1

    if(method == "low"):
        mask = mask
    # 高通滤波则取反
    elif(method == "high"):
        mask = 1 - mask
    # 其余则报错
    else:
        return "Error"

    # 逆中心平移
    reverse_fouier_center_r = fouier_center_r * mask
    reverse_fouier_center_g = fouier_center_g * mask
    reverse_fouier_center_b = fouier_center_b * mask
    reverse_fouier_r = np.fft.ifftshift(reverse_fouier_center_r)
    reverse_fouier_g = np.fft.ifftshift(reverse_fouier_center_g)
    reverse_fouier_b = np.fft.ifftshift(reverse_fouier_center_b)

    # 逆傅里叶变换
    img_after_process = np.zeros([height, width, channels], dtype='uint8')
    img_after_process[:, :, 0] = np.abs(np.fft.ifft2(reverse_fouier_r))
    img_after_process[:, :, 1] = np.abs(np.fft.ifft2(reverse_fouier_g))
    img_after_process[:, :, 2] = np.abs(np.fft.ifft2(reverse_fouier_b))

    return img_after_process


# 空域变换
img_space_blur = np.zeros([height, width, 3], dtype='uint8')
img_space_sharpen = np.zeros([height, width, 3], dtype='uint8')
img_space_blur[:, :, :] = space_average_blur(img_rgb, 5)
img_space_sharpen[:, :, :] = space_sharpen(img_space_blur)

# 频域变换
img_frequency_blur = np.zeros([height, width, 3], dtype='uint8')
img_frequency_sharpen = np.zeros([height, width, 3], dtype='uint8')
img_frequency_blur[:, :, :] = frequency_change(img_rgb, 6, "low")
img_frequency_sharpen[:, :, :] = frequency_change(img_frequency_blur, 6, "high")


# 显示图形
titles = ['Source Image',
          'Space Blur Image', 'Space Sharpen Image',
          'Frequency Blur Image', 'Frequency Sharpen Image']
images = [img_rgb,
          img_space_blur, img_space_sharpen,
          img_frequency_blur, img_frequency_sharpen]
for i in range(len(images)):
    row = 2
    plt.subplot(row, int((len(images) + 1) / row), i + 1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()




