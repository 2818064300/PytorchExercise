import time
import torch
import utils.Plot as plt
from MnistDataset import MnistDataset
from Classifier import Classifier
from LinearRegressionModel import LinearRegressionModel


def test_Classifier():
    start = time.time()
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    else:
        print('cpu')
    # 训练用数据集
    mnist_dataset = MnistDataset('mnist_dataset/mnist_train.csv')
    # 分类器
    C = Classifier()
    epochs = 5
    for i in range(epochs):
        print('training epoch', i + 1, 'of', epochs)
        for label, image_data_tensor, target_tensor in mnist_dataset:
            C.train(image_data_tensor, target_tensor)
    # 绘制损失函数
    plt.plot_progress(C.progress)
    # 绘制概率直方图
    C.plot_hist(1)
    end = time.time()
    print("运行时间 ", end - start)


def test_LinearRegressionModel():
    x_values = [i for i in range(11)]
    x_train = torch.FloatTensor(x_values).reshape(-1, 1)
    y_values = [i * 2 + 1 for i in x_values]
    y_train = torch.FloatTensor(y_values).reshape(-1, 1)
    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)
    epochs = 1000
    for e in range(epochs):
        e += 1
        model.optimiser.zero_grad()
        outputs = model(x_train)
        loss = model.loss_function(outputs, y_train)
        loss.backward()
        model.optimiser.step()
        model.progress.append(loss.item())
        if e % 50 == 0:
            print("epochs {}, loss {}".format(e, loss.item()))
    torch.save(model.state_dict(), 'model/model.pkl')
    plt.plot_progress(model.progress)


# 框架测试
if __name__ == '__main__':
    pass
    # test_LinearRegressionModel()
