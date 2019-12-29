import torch

class AlexNet(torch.nn.Module):
    '''
    1、Relu激活函数
    2、Dropout避免过拟合
    3、使用最大池化，避免平均池化的模糊化效果
    4、提出步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性
    5、提出LRN层
    6、CUDA-GPU加速
    7、数据增强

    输入尺寸：227*227*1
    卷积层：5个
    降采样层(池化层)：3个
    全连接层：2个
    输出层：1个。1000个类别
    '''
    def __init__(self, in_channel=1, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = torch.nn.Sequential(

            # Input: 227*227*1
            # C1: 55*55*96 (11*11*3 kernel 4 stride)
            torch.nn.Conv2d(in_channel, 96, 11, 4),
            torch.nn.ReLU(),
            # M1: 27*27*96 (3*3 kernel 2 stride)
            torch.nn.MaxPool2d(3, 2),

            # C2: 256*27*27（5*5 kernel 2 padding）
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            # M2: 256*13*13 (3*3 kernel 2 stride)
            torch.nn.MaxPool2d(3, 2),

            # C3: 384*13*13 (3*3 kernel 1 padding)
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),

            # C4: 384*13*13 (3*3 kernel 1 padding)
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),

            # C5: 256*13*13 (3*3 kernel 1 padding)
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            # M3: 256*6*6 (3*3 kernel 2 stride)
            torch.nn.MaxPool2d(3, 2),
        )

        self.classifier = torch.nn.Sequential(

            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256*6*6, 4096),
            torch.nn.ReLU(),

            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),

            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

'''
测试
'''
# import numpy as np
# import torch
# img = np.random.randint(0, 255, size=(1, 1, 227, 227))
# print(img.shape)
# img = torch.tensor(img, dtype=torch.float32)
# net = AlexNet()
# output = net(img)
# print(output)