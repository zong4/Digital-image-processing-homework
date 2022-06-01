# -*- coding : utf-8 -*-
# @Time      : 2022/4/23 20:02
# @Author    : Zong
# @File      : train.py
# @Software  : PyCharm
# @Function  : 模型训练
# @ChangeLog :


import torch
from py_path.dataset import train_dataloader, test_dataloader
import config
from py_path.model import MODEL


# 调取环境
device = config.DEVICE

# 设置优化器
if config.OPTIM == "SGD":
    optim = torch.optim.SGD(MODEL.parameters(), lr = config.LEARN_RATE)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0


for step in range(config.EPOCH):
    # 显示运行轮数
    print("-------第 {} 轮训练开始-------".format(step + 1))

    # 训练开始
    MODEL.train()

    for data in train_dataloader:
        imgs, masks = data

        # 转为double型
        imgs = imgs.to(torch.float32)
        imgs = imgs.to(device)
        masks = masks.to(torch.float32)
        masks = masks.to(device)
        # print(imgs.shape)
        # print(masks.shape)
        # with torch.no_grad():

        # 模型输出
        outputs = MODEL(imgs)
        # output_predictions = outputs.argmax(0)
        # print(outputs["out"].shape)
        # print(masks.shape)
        inputs = outputs["out"]
        target = masks.long()

        # 计算损失
        loss = config.LOSS_FUN(inputs, target)

        # 优化器优化
        optim.zero_grad()
        loss.backward()
        optim.step()

        # 训练次数 + 1
        total_train_step = total_train_step + 1

        # 训练PRINT_INTERVAL次输出一次
        if total_train_step % config.PRINT_INTERVAL == 0:
            print("训练次数：{}, Loss: {}, Image size: (H: {}, W: {}), Batch size: {}".format(total_train_step, loss.item(), config.HEIGHT, config.WIDTH, config.TRAIN_BATCH_SIZE))


    # 测试开始
    print("-------第 {} 轮测试开始-------".format(step + 1))
    MODEL.eval()
    total_test_loss = 0
    # total_accuracy = 0

    # 无梯度
    with torch.no_grad():
        for data in test_dataloader:
            imgs, masks = data
            imgs = imgs.to(torch.float32)
            imgs = imgs.to(device)
            masks = masks.to(torch.float32)
            masks = masks.to(device)

            # 模型输出
            outputs = MODEL(imgs)

            # 去除输出的第三维度
            inputs = outputs["out"]
            target = masks.long()

            # 计算loss
            loss = config.LOSS_FUN(inputs, target)

            # 测试次数 + 1
            total_test_step = total_test_step + 1
            # 训练PRINT_INTERVAL次输出一次
            if total_test_step % config.PRINT_INTERVAL == 0:
                print("测试次数：{}, Loss: {}, Image size: (H: {}, W: {}), Batch size: {}".format(total_test_step, loss.item(), config.HEIGHT, config.WIDTH, config.TEST_BATCH_SIZE))


    # 保存模型
    torch.save(MODEL, config.MODEL_STORE_DIR + "model_{}.pth".format(step + 1))
    print("-------模型已保存-------")


print("-------训练结束-------")