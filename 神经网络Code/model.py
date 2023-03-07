import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import transforms


# class Hao(nn.Module):
#
#
#
#     '''MNIST'''
#     def __init__(self):
#         super(Hao, self).__init__()
#         # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
#         # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数（即用了几个卷积核），第三个参数指卷积核的大小
#         self.conv1 = nn.Conv2d(3, 10,kernel_size=(5,5))  # 因为图像为黑白的，所以输入通道为1,此时输出数据大小变为28-5+1=24.所以batchx1x28x28 -> batchx10x24x24
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=(5,5))  # 第一个卷积层的输出通道数等于第二个卷积层是输入通道数。
#         # self.conv2_drop = nn.Dropout2d()  # 在前向传播时，让某个神经元的激活值以一定的概率p停止工作，可以使模型泛化性更强，因为它不会太依赖某些局部的特征
#         self.fc1 = nn.Linear(500, 50)  # 由于下部分前向传播处理后，输出数据为20x4x4=320，传递给全连接层。# 输入通道数是320，输出通道数是50
#         self.fc2 = nn.Linear(50, 10)  # 输入通道数是50，输出通道数是10，（即10分类（数字1-9），最后结果需要分类为几个就是几个输出通道数）。全连接层（Linear）：y=x乘A的转置+b
#
#
#     def forward(self, x):
#         a =x
#         x1 = self.conv1(x)
#         x2 = torch.max_pool2d(x1, 2)
#         x3 = F.relu(x2)  # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半，步长为2）（激活函数ReLU不改变形状）
#         # x = F.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)),2))  # 此时输出数据大小变为12-5+1=8（卷积核大小为5）（2*2的池化层会减半）。所以 batchx10x12x12 -> batchx20x4x4。
#         x4 = self.conv2(x3)
#         x5 = torch.max_pool2d(x4,2)
#         x6 = F.relu(x5)
#
#         x7 = nn.Flatten()(x6)  # batch*20*4*4 -> batch*320
#         x8 = F.relu(self.fc1(x7))  # 进入全连接层
#         # x = F.dropout(x, training=self.training)  # 减少遇到过拟合问题，dropout层是一个很好的规范模型。
#         out = self.fc2(x8)
#         return out


class Hao(nn.Module):


    def __init__(self):
        super(Hao,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)

        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)

        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)

        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(2048,1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024,10)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)

        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)

        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

# if __name__ == '__main__':
#     hao =Hao()
#     input = torch.ones((1,1,32,32))
#     output = hao(input)
#     print(output.shape)