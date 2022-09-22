import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import Utils.Plot as plt
from CNNDiscriminator import Discriminator
from CNNGenerator import Generator
from Mnist.MnistDataset import MnistDataset


# 生成随机种子
def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


# 框架测试
if __name__ == '__main__':
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # D = Discriminator()
    # G = Generator()
    # # D.to(device)
    # # G.to(device)
    # mnist_dataset = MnistDataset('../../Mnist/mnist_dataset/mnist_train.csv')
    # for label, image_data_tensor, target_tensor in mnist_dataset:
    #     D.train(image_data_tensor.reshape(1, 1, 28, 28), torch.FloatTensor([1.0]))
    #     D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
    #     G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))
    # 下载MNIST数据集
    datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor, download=True)
