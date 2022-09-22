import pandas
import matplotlib.pyplot as plt


# 绘制损失函数图像
def plot_progress(progress):
    df = pandas.DataFrame(progress, columns=['loss'])
    df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
    plt.show()


# 绘制图像
def plot_1010(data):
    plt.figure(figsize=(16, 8))
    plt.imshow(data, interpolation='None', cmap='Blues')
    plt.show()
