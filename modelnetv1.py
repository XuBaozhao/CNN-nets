import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        # 深度卷积，通道数不变，用于缩小特征图大小
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        # 逐点卷积，用于增大通道数
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNetV1(nn.Module):
    cfg = [
        64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024
    ]
    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        # 首先是一个标准卷积
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 然后堆叠深度可分离卷积
        self.layers = self._make_layers(in_planes=32)

        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        laters = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            laters.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*laters)

    def forward(self, x):
        # 一个普通卷积
        out = F.relu(self.bn1(self.conv1(x)))
        # 叠加深度可分离卷积
        out = self.layers(out)
        # 平均池化层会将feature变成1x1
        out = F.avg_pool2d(out, 7)
        # 展平
        out = out.view(out.size(0), -1)
        # 全连接层
        out = self.linear(out)
        # softmax层
        output = F.softmax(out, dim=1)
        return output

'''测试'''
# def test():
#     net = MobileNetV1()
#     x = torch.randn(1, 3, 224, 224)
#     y = net(x)
#     print(y.size())
#     print(y)
#     print(torch.max(y,dim=1))
#
# test()
# net = MobileNet()
# print(net)