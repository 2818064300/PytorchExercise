import torch
import torch.nn as nn


# 定义神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 输入大小 (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 灰度图
                out_channels=16,  # 要得到几多少个特征图
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),  # 输出的特征图为 (16, 28, 28)
            nn.ReLU(),  # relu层
            nn.MaxPool2d(kernel_size=2),  # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 14, 14)
            nn.ReLU(),  # relu层
            nn.MaxPool2d(2),  # 输出 (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层得到的结果

        # 定义学习率
        self.lr = 0.001

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()
        # 优化器
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten操作，结果为：(batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output
