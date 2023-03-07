import torch
import torch.nn as nn

# num_features - num_features from an expected input of size:batch_size*num_features*height*width
# eps:default:1e-5 (公式中为数值稳定性加到分母上的值)
# momentum:动量参数，用于running_mean and running_var计算的值，default：0.1
m = nn.BatchNorm2d(1, affine=True)  # affine参数设为True表示weight和bias将被使用
Input = torch.randn(1, 1, 3, 4)
output = m(Input)

'''利用公式'''
Means = torch.mean(Input)
Vars = torch.var(Input, False)  # 表示贝塞尔校正不会被使用
y = (Input - Means) / (torch.sqrt(Vars) + 1e-5) * m.weight + m.bias

print('输入:', Input)
print(m.weight)
print(m.bias)
print('nn.BN:', output)
print('公式:', y)
