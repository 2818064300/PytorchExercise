import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from CNN import CNN

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

if __name__ == '__main__':
    # 定义超参数
    input_size = 28  # 图像的总尺寸28*28
    num_classes = 10  # 标签的种类数
    num_epochs = 3  # 训练的总循环周期
    batch_size = 64  # 一个撮（批次）的大小，64张图片

    # 训练集
    train_dataset = datasets.MNIST(root='./dataset',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    # 测试集
    test_dataset = datasets.MNIST(root='./dataset',
                                  train=False,
                                  transform=transforms.ToTensor())

    # 构建batch数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    # 实例化
    net = CNN()

    # 开始训练循环
    for epoch in range(num_epochs):
        # 当前epoch的结果保存下来
        train_rights = []

        for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
            net.train()
            output = net(data)
            loss = net.loss_function(output, target)
            net.optimiser.zero_grad()
            loss.backward()
            net.optimiser.step()
            right = accuracy(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:

                net.eval()
                val_rights = []

                for (data, target) in test_loader:
                    output = net(data)
                    right = accuracy(output, target)
                    val_rights.append(right)

                # 准确率计算
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                print(
                    '当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.data,
                               100. * train_r[0].numpy() / train_r[1],
                               100. * val_r[0].numpy() / val_r[1]))
