import torch.nn as nn

# 配置文件一般以 .cfg 为后缀
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    '''
    conv+relu+batch_normalize+pooling 卷积一般这么操作
    全连接层可以使用：Linear+relu+dropouts

    输入尺寸：224*224*1
    卷积层：n
    降采样层(池化层)：n
    全连接层：3
    输出层：1个。n个类别
    '''
    def __init__(self, vgg_name, in_channel=1, num_classes=10):
        super(VGG, self).__init__()

        self.features = self._make_layers(in_channel, cfg[vgg_name])

        self.classifier = nn.Sequential(

            # inpalce=True,可以节省运算内存，不用多储存变量
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_channel, cfg):
        layers = []
        in_channels = in_channel

        print(cfg)

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # print('x is number')
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)
                           ]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11():
    return VGG('VGG11')

def VGG13():
    return VGG('VGG13')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')

'''
测试
'''
# import numpy as np
# import torch
# img = np.random.randint(0,255,size=(1, 1, 224, 224))
# img = torch.tensor(img, dtype=torch.float32)
# net = VGG19()
# output = net(img)
# print(output)