import pandas
import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt


# 手写数字数据集
class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0

        return label, image_values, target

    def plot_image(self, index):
        arr = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title('lebel = ' + str(self.data_df.iloc[index, 0]))
        plt.imshow(arr, cmap='Blues', interpolation='None')
        plt.show()
