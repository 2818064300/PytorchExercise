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

        # 定义神经网络层
        # self.model = nn.Sequential(
        #     nn.Linear(1, 200),
        #     nn.Sigmoid(),
        #     nn.Linear(200, 784),
        #     nn.Sigmoid()
        # )
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid(),
        )
        # 定义学习率
        self.lr = 0.0001

        # 创建优化器 使用随机梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        # 计数器
        self.counter = 0
        self.progress = []

    def forward(self, image_tensor, label_tensor):
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)

    # 训练
    def train(self, D, inputs, label_tensor, targets):
        # 计算网络的输出
        g_outputs = self.forward(inputs, label_tensor)
        # 鉴别器的输出
        d_outputs = D.forward(g_outputs, label_tensor)
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
