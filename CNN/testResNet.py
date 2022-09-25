import torch
import torchvision.models as models

if __name__ == '__main__':
    model = models.resnet152()
    print(model)