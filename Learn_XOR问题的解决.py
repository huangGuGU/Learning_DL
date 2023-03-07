import torch
import torch.nn as nn

input = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
label = torch.tensor([0,1,1,0])


class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.hidden1 = nn.Linear(2, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.hidden3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()



    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))

        return x


net = XOR()
optim = torch.optim.Adam(net.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
total_accuracy = 0
# loss_function = nn.MSELoss()
for epoch in range(500):
    output = net(input)
    loss = loss_function(output, label)
    optim.zero_grad()  # 梯度调为0
    loss.backward()  # 反向传播求出每个节点的梯度
    optim.step()  # 对每个参数进行调优
    print(output.argmax(1))


    print('running_loss', loss.item())


