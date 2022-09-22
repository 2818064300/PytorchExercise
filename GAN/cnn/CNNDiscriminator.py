import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


# 鉴别器
class Discriminator(nn.Module):
    def __init__(self):
        # 初始化Pytorch父类
        super().__init__()

        # 定义卷积神经网络
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(10),

            nn.Conv2d(10, 10, kernel_size=3, stride=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(10),

            View(250),
            nn.Linear(250, 1),
            nn.Sigmoid()
        )
        # 定义学习率
        self.lr = 0.0001

        # 定义损失函数
        # self.loss_function = nn.MSELoss()
        self.loss_function = nn.BCELoss()

        # 创建优化器 使用随机梯度下降
        self.optimiser = torch.optim.Adam(self.parameters())

        # 计数器
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    # 训练
    def train(self, inputs, targets):
        # 计算网络的输出
        outputs = self.forward(inputs)
        # 计算损失值
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print('counter = ', self.counter)

        # 归零梯度
        self.optimiser.zero_grad()
        # 反向传播
        loss.backward()
        # 更新权重
        self.optimiser.step()
