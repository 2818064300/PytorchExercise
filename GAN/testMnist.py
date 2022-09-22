import random

import torch
from MnistDiscriminator import Discriminator
from MnistGenerator import Generator
import utils.Plot as plt
import matplotlib.pyplot
from Mnist.MnistDataset import MnistDataset


# 生成随机数据源
def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


# 生成随机种子
def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


def show():
    f, axarr = matplotlib.pyplot.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(100))
            img = output.detach().numpy().reshape(28, 28)
            axarr[i, j].imshow(img, interpolation='none', cmap='Blues')


def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0, size - 1)
    label_tensor[random_idx] = 1.0
    return label_tensor


def plot_images(label):
    label_tensor = torch.zeros((10))
    label_tensor[label] = 1.0
    f, axarr = matplotlib.pyplot.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            axarr[i, j].imshow(
                G.forward(generate_random_seed(100), label_tensor).detach().cpu().numpy().reshape(28, 28),
                interpolation='none', cmap='Blues')
    matplotlib.pyplot.show()


# 训练神经网络数据集
def train(epochs):
    D = Discriminator()
    G = Generator()
    mnist_dataset = MnistDataset('D:\mnist_dataset\mnist_train.csv')
    for e in range(epochs):
        print("世代为: " + str(e + 1))
        for label, image_data_tensor, label_tensor in mnist_dataset:
            D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))
            random_label = generate_random_one_hot(10)
            D.train(G.forward(generate_random_seed(100), random_label).detach(), random_label, torch.FloatTensor([0.0]))
            random_label = generate_random_one_hot(10)
            G.train(D, generate_random_seed(100), random_label, torch.FloatTensor([1.0]))


# 框架测试
if __name__ == '__main__':
    # D = Discriminator()
    # G = Generator()
    # epochs = 3
    # print("开始训练,世代为: " + str(epochs))
    # mnist_dataset = MnistDataset('../Mnist/mnist_dataset/mnist_train.csv')
    # for e in range(epochs):
    #     print("世代为: " + str(e))
    #     for label, image_data_tensor, target_tensor in mnist_dataset:
    #         D.train(image_data_tensor, torch.FloatTensor([1.0]))
    #         D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
    #         G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))
    # plt.plot_progress(D.progress)
    # show()
    train(1)

    # 加载模型
    D = torch.load('./model/mnist_D1.pt')
    G = torch.load('./model/mnist_G1.pt')
    plot_images(1)
