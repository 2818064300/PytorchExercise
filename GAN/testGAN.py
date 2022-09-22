import torch
import numpy
import random
from Discriminator import Discriminator
from Generator import Generator
import utils.Plot as plt


# 生成真实数据源
def generate_real():
    real_data = torch.FloatTensor(
        [random.uniform(0.8, 1.0), random.uniform(0.0, 0.2), random.uniform(0.8, 1.0), random.uniform(0.0, 0.2)])
    return real_data


# 生成随机数据源
def generate_random(size):
    random_data = torch.rand(size)
    return random_data


# 框架测试
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    else:
        print('cpu')

    # 鉴别器
    D = Discriminator()
    # 生成器
    G = Generator()

    image_list = []

    for i in range(10000):
        # 真实数据
        D.train(generate_real(), torch.FloatTensor([1.0]))
        # 随机数据
        D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
        # 训练生成器
        G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
        if i % 1000 == 0:
            image_list.append(G.forward(torch.FloatTensor([0.5])).detach().numpy())
    plt.plot_progress(D.progress)
    plt.plot_1010(numpy.array(image_list).T)
    print("训练结束")
