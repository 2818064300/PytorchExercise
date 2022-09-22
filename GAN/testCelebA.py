import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import random
import numpy

dataset = datasets.ImageFolder('./dataset')
plt.imshow(dataset[0][0])
plt.title('label 1')
plt.show()
plt.imshow(dataset[1][0])
plt.title('label 2')
plt.show()

if __name__ == '__main__':
    mnist_dataset = datasets.CelebA(root='.', download=True)

