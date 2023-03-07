from torch.utils.data import DataLoader
from model import *
from DataSet import *
import numpy as np
from tensorboardX import SummaryWriter
from loss_use import FocalLoss


import torchvision

save_test_path = 'testlog.txt'
dev = torch.device('cpu')


def main():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./Dataset/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])),

        batch_size=64, shuffle=True)
    # 测试集数据
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./Dataset/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                             ])),
        batch_size=64, shuffle=True)

    tb_writer = SummaryWriter(log_dir='tb')
    test_len = test_loader.sampler.num_samples
    train_len = train_loader.sampler.num_samples


    '''创建网络模型'''
    hao = Hao()
    # hao.to(dev)
    # model_structure_parameters(Hao())
    init_img = torch.zeros(1, 3, 32, 32)
    # tb_writer.add_graph(hao, init_img)
    '''loss'''
    # loss_fn = FocalLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hao.parameters(), lr=0.001, )

    epoch = 10

    for i in range(epoch):
        print('No.{} training'.format(i))
        total_loss = 0
        index = 0
        train_total_accuracy = 0
        for index, data in enumerate(train_loader):
            imgs, targets = data

            if torch.cuda.is_available():
                imgs = imgs.to(dev)
                targets = targets.to(dev)
            output = hao(imgs)
            accuracy = ((output.argmax(1) == targets).sum())
            train_total_accuracy += accuracy
            loss = loss_fn(output, targets)
            optimizer.zero_grad()

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        loss_avg = total_loss/(index+1)
        print('train accuracy:{}, total train loss:{}'.format(round(float(train_total_accuracy / train_len), 3),
                                                            round(float(loss_avg), 4)))








        tags = ['train_loss', 'test_loss', 'accuracy']
        tb_writer.add_scalar(tags[0], loss.item(), epoch)

        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            index = 0
            for index, data in enumerate(test_loader):
                imgs, targets = data
                if torch.cuda.is_available():
                    imgs = imgs.to(dev)
                    targets = targets.to(dev)
                output = hao(imgs)
                test_loss = loss_fn(output, targets)
                total_test_loss += test_loss.item()
                accuracy = ((output.argmax(1) == targets).sum())
                total_accuracy += accuracy

            with open(save_test_path, 'a') as f:
                f.write('test epoch {} loss {} accuracy {}\n'.format(i, total_test_loss / train_len,
                                                                     total_accuracy / train_len))
                torch.save(hao, 'logs/Hao{}.pth'.format(i))
                # print('model is saved')
        print('test accuracy:{}, total test loss:{}'.format(round(float(total_accuracy / test_len), 3),
                                                            round(float(total_test_loss/(index+1)), 4)))


        tb_writer.add_scalar(tags[1], test_loss.item(), epoch)
        tb_writer.add_scalar(tags[2], total_accuracy / test_len, epoch)
    tb_writer.close()
    # torch.save(hao.state_dict(),'hao_Method_1.pth')


# def model_structure_parameters(m):
#     blank = ' '
#     print('-' * 119)
#     print('|' + ' ' * 30 + 'weight name' + ' ' * 30 + '|' + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' + ' ' * 3 +
#           'number' + ' ' * 3 + '|')
#     print('-' * 119)
#     num_para = 0
#
#     for index, (key, w_variable) in enumerate(m.named_parameters()):
#         if len(key) <= 69:
#             key = key + (69 - len(key)) * blank
#         shape = str(w_variable.shape)
#         if len(shape) <= 30:
#             shape = shape + (30 - len(shape)) * blank
#         each_para = 1
#         for k in w_variable.shape:
#             each_para *= k
#         num_para += each_para
#         str_num = str(each_para)
#         if len(str_num) <= 10:
#             str_num = str_num + (10 - len(str_num)) * blank
#         print('| {} | {} | {} |'.format(key, shape, str_num))
#
#     print('-' * 119)
#     print('The total number of parameters: ' + str(num_para))
#     print('The parameters of Model {}: {:.2f}M'.format(type(m).__name__, num_para / 1e6))
#     print('-' * 119)


if __name__ == '__main__':
    main()

    # writer = SummaryWriter(log_dir='tb')
    # model = Hao()
    # # 将模型写入tensorboard
    # init_img = torch.zeros((1, 1, 28, 28))
    # writer.add_graph(model, init_img)
    # import random
    # tags = ["train_loss", "accuracy", "learning_rate"]
    # for epoch in range(30):
    #     mean_loss = random.randint(0,10)
    #     writer.add_scalar("train_loss", mean_loss, epoch)
    #
    # writer.add_histogram(tag="conv1",values=model.conv1.weight,global_step=epoch)
    # writer.close()
