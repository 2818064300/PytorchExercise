import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import copy
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image


# 展示数据
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


# 展示图片
def show():
    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2

    dataiter = iter(dataloaders['valid'])
    inputs, classes = dataiter.next()

    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
        plt.imshow(im_convert(inputs[idx]))
    plt.show()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        # 更改全连接层
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))
        input_size = 224
    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename='checkpoint.pth'):
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


# 主程序
if __name__ == '__main__':

    # 数据集路径
    data_dir = './dataset/flower_data/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # 数据预处理,数据集增强
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                                     transforms.CenterCrop(224),  # 从中心开始裁剪
                                     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                                     transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                     transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                     # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                     transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                                     ]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
    }

    batch_size = 8

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    # 读取类别
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    '''
    使用resnet网络进行训练
    可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    '''
    model_name = 'resnet'

    # 是否用人家训练好的特征来做,是否更新梯度
    feature_extract = True

    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    # 使用GPU进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet152()

    model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

    # GPU计算
    model_ft = model_ft.to(device)

    #  模型保存
    filename = 'checkpoint.pth'

    # 是否训练所有层
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # 优化器设置
    optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减成原来的1/10
    # 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
    criterion = nn.NLLLoss()

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                                criterion, optimizer_ft,
                                                                                                num_epochs=20,
                                                                                                is_inception=(
                                                                                                        model_name == "inception"))

    for param in model_ft.parameters():
        param.requires_grad = True

    # 再继续训练所有的参数，学习率调小一点
    optimizer = optim.Adam(params_to_update, lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 损失函数
    criterion = nn.NLLLoss()

    # Load the checkpoint

    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model_ft.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # model_ft.class_to_idx = checkpoint['mapping']

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                                criterion, optimizer,
                                                                                                num_epochs=10,
                                                                                                is_inception=(
                                                                                                        model_name == "inception"))

    model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

    # GPU模式
    model_ft = model_ft.to(device)

    #  保存文件的名字
    filename = 'seriouscheckpoint.pth'

    # 加载模型
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model_ft.load_state_dict(checkpoint['state_dict'])

    image_path = 'image_06621.jpg'
    img = process_image(image_path)
    imshow(img)

    # 得到一个batch的测试数据
    dataiter = iter(dataloaders['valid'])
    images, labels = dataiter.next()

    model_ft.eval()

    if train_on_gpu:
        output = model_ft(images.cuda())
    else:
        output = model_ft(images)

    _, preds_tensor = torch.max(output, 1)

    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

    fig = plt.figure(figsize=(20, 20))
    columns = 4
    rows = 2

    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        plt.imshow(im_convert(images[idx]))
        ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                     color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
    plt.show()
