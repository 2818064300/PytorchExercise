import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt


# 建立线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 线性回归层
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.progress = []

    def forward(self, x):
        out = self.linear(x)
        return out

