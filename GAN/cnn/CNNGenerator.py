import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


# 生成器
class Generator(nn.Module):
    def __init__(self):
        # 初始化Pytorch父类
        super().__init__()

        # 定义卷积神经网络
        self.model = nn.Sequential(
            nn.Linear(100, 10 * 5 * 5),
            View((1, 10, 5, 5)),
            nn.ConvTranspose2d(10, 10, kernel_size=3, stride=2),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.02),
            nn.ConvTranspose2d(10, 1, kernel_size=5, stride=2),
            nn.Sigmoid()
        )

        # 定义学习率
        self.lr = 0.0001

        # 创建优化器 使用随机梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        # 计数器
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    # 训练
    def train(self, D, inputs, targets):
        # 计算网络的输出
        g_outputs = self.forward(inputs)
        # 鉴别器的输出
        d_outputs = D.forward(g_outputs)
        # 计算损失值
        loss = D.loss_function(d_outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        # 归零梯度
        self.optimiser.zero_grad()
        # 反向传播
        loss.backward()
        # 更新权重
        self.optimiser.step()
