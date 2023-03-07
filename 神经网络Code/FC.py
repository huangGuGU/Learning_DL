import torch
from torch import nn
import torch.nn.functional as F



class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(24 * 24, 24 * 24)
        self.fc2 = nn.Linear(24 * 24, 24 * 24)
        self.fc3 = nn.Linear(24 * 24, 24 * 24)
        self.fc4 = nn.Linear(24 * 24, 24 * 24)


    def forward(self, x):

        x1 = x.view(x.shape[0],x.shape[1],-1)

        layer1 = self.fc1(x1)
        layer2 = self.fc2(layer1)
        layer3 = self.fc3(layer2)
        layer4 = self.fc4(layer3)

        out = layer4.view(x.shape)

        return out

def model_structure_parameters(m):
    blank = ' '
    print('-' * 119)
    print('|' + ' ' * 30 + 'weight name' + ' ' * 30 + '|' + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' + ' ' * 3 +
          'number' + ' ' * 3 + '|')
    print('-' * 119)
    num_para = 0

    for index, (key, w_variable) in enumerate(m.named_parameters()):
        if len(key) <= 69:
            key = key + (69 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank
        print('| {} | {} | {} |'.format(key, shape, str_num))

    print('-' * 119)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:.2f}M'.format(type(m).__name__, num_para / 1e6))
    print('-' * 119)


if __name__ == '__main__':
    fcn =FCN()
    model_structure_parameters(fcn)
    input = torch.rand((4,1,24,24))
    output = fcn(input)
    print(output.shape)