from vgg import VGG

if __name__ == '__main__':
    vgg = VGG(21).cuda()

    # 输出VGGNet卷积层
    print(vgg.features)

    # 输出VGGNet全连接层
    print(vgg.classifier)
