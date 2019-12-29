import torch
import torch.nn as nn
'''
理解案例
Input: 1*32*32

Conv1:
C1: 6*28*28 (5*5 kernel)
ReLu
S2: 6*14*14 (2*2 kernel, stride 2)

Conv2:
C3: 16*10*10 (5*5 kernel)
ReLu
S4: 16*5*5 (2*2 kernel, stride 2)

Conv3: 120*1*1 (5*5 kernel)

F6: 84

ReLu

F7: 10
'''
class Lenet_5(nn.Module):
    '''
    1、卷积、池化、非线性激活函数
    2、采用卷积提取空间特征
    3、降采样的平均池化层
    4、双曲正切Tanh或S型Sigmoid的激活函数
    5、多层感知机MLP作为最后的分类器
    6、层与层之间的稀疏连接减少计算复杂度

    输入尺寸：32*32
    卷积层：2个
    降采样层(池化层)：2个
    全连接层：2个
    输出层：1个。10个类别（数字0-9的概率）
    '''
    def __init__(self, in_channel=1, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, 5),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(120*1*1, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),

            # 在pytorch中若模型使用CrossEntropyLoss这个loss函数，则不应该在最后一层再使用softmax进行激活。
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(input.size(0), -1)
        print(output.shape)
        output = self.fc(output)
        return output

'''
测试
'''
# import numpy as np
# import torch
# img = np.random.randint(0, 255, size=(1, 1, 32, 32))
# print(img.shape)
# img = torch.tensor(img, dtype=torch.float32)
# net = Lenet_5()
# output = net(img)
# print(output)

