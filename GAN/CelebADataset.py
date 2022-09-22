import pandas
import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import torchvision.datasets as datasets


class CelebADataset(Dataset):
    def __init__(self, path):
        self.dataset = datasets.ImageFolder(path)
        pass

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
