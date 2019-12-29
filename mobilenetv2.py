import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.cfg = x
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(self.cfg[0], self.cfg[1], kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.cfg[1]),
            nn.ReLU6()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(self.cfg[2], self.cfg[3], kernel_size=3, padding=1, stride=self.cfg[6]),
            nn.BatchNorm2d(self.cfg[3]),
            nn.ReLU6()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(self.cfg[4], self.cfg[5], kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.cfg[5]),
            nn.ReLU6()
        )

    def forward(self, x):
        if self.cfg[7] == 1:
            residual = x
        output = self.conv1x1_1(x)
        output = self.conv3x3(output)
        output = self.conv1x1_2(output)
        if self.cfg[7] == 1:
            output += residual
        return output

class MobileNetV2(nn.Module):
    cfg = [
        # in-out-in-out-in-out-stride-residual
        (32, 32, 32, 32, 32, 16, 1, 0),
        (16, 96, 96, 96, 96, 24, 2, 0),
        (24, 144, 144, 144, 144, 24, 1, 1), # add1
        (24, 144, 144, 144, 144, 32, 2, 0),
        (32, 192, 192, 192, 192, 32, 1, 1), # add2
        (32, 192, 192, 192, 192, 32, 1, 1), # add3
        (32, 192, 192, 192, 192, 64, 1, 0),
        (64, 384, 384, 384, 384, 64, 1, 1), # add4
        (64, 384, 384, 384, 384, 64, 1, 1), # add5
        (64, 384, 384, 384, 384, 64, 1, 1), # add6
        (64, 384, 384, 384, 384, 96, 2, 0),
        (96, 576, 576, 576, 576, 96, 1, 1), # add7
        (96, 576, 576, 576, 576, 96, 1, 1), # add8
        (96, 576, 576, 576, 576, 160, 2, 0),
        (160, 960, 960, 960, 960, 160, 1, 1),  # add9
        (160, 960, 960, 960, 960, 160, 1, 1),  # add10
        (160, 960, 960, 960, 960, 320, 1, 0),  # add11
    ]
    def __init__(self, in_channel=3, NUM_CLASSES=10):
        super().__init__()

        # 首先一个普通卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        # 深度卷积可分离+Inverted residuals
        self.layers = self._make_layers()
        # 将逐点卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )
        # 全局平均池化，将图像变成1x1大小
        self.pool = nn.AvgPool2d(kernel_size=7)
        # 最后为全连接
        self.linear = nn.Sequential(
            nn.Linear(1280, NUM_CLASSES)
        )

    def _make_layers(self):
        layers = []
        for x in self.cfg:
            layers.append(Bottleneck(x))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layers(output)
        output = self.conv2(output)
        output = self.pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

'''测试'''
# def test():
#     net = MobileNetV2()
#     x = torch.randn(1, 3, 224, 224)
#     y = net(x)
#     print(y.size())
#
# test()
# net = MobileNetV2()
# print(net)

