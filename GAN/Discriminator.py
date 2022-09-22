import torch
import torch.nn as nn


# 鉴别器
class Discriminator(nn.Module):
    def __init__(self):
        # 初始化Pytorch父类
        super().__init__()

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        # 定义学习率
        self.lr = 0.01

        # 定义神经网络层
        self.loss_function = nn.MSELoss()

        # 创建优化器 使用随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)

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

