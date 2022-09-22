import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt

from MnistDataset import MnistDataset


# 分类器
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        # 更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print('counter = ', self.counter)

    def plot_hist(self, index):
        mnist_test_dataset = MnistDataset('mnist_dataset/mnist_test.csv')
        record = index
        img_data = mnist_test_dataset[record][1]
        output = self.forward(img_data)
        pandas.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0, 1))
        plt.show()
        mnist_test_dataset.plot_image(index)
