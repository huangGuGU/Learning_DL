from pylab import *  # 支持中文
import os



path = 'loss_accuracy_log'
save_path = 'loss_accuracy_img'


def makeachart(file_path,model_name):
    t = '30'
    x = []
    y = []
    z = []


    '''test epoch 0 loss 15.921034216880798 accuracy 86.9000015258789'''

    with open(file_path) as f:
        for c in f.readlines():
            s = c.split(' ')

            loss = s[4]
            epoch = s[2]
            accuracy = float(s[6])
            accuracy = round(accuracy,3)
            x.append(epoch)
            y.append(float(loss))

            z.append(accuracy)
    n = range(len(x))

    # plt.plot(n, y, marker='', mec='r', mfc='w', label='loss')
    plt.plot(n, z, marker='', mec='r', mfc='w', label='accuracy')



    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("epoch_num")
    plt.ylabel("accuracy")  # Y轴标签
    plt.title(model_name+"_accuracy")  # 标题

    Save_path = os.path.join(save_path,model_name+'.png')
    plt.savefig(Save_path)

    plt.show()


def replace_name(file_path):
    with open (file_path) as f:
        for c in f.readlines():
            c = c.replace(':',' ')
            with open(file_path,'a') as f:
                f.write(c)
    print('done')




filelist = os.listdir(path)
for model_name in filelist:
    file_path = os.path.join(path,model_name)
    # replace_name(file_path)
    makeachart(file_path,model_name[:-4])



















def main():
    with open(path) as f:
        for i in f.readlines():


                i = i.replace(':', ' ')

                print(i)
                with open(path,'a') as f:
                    f.write(i)





# main()
'''写入tensorboard'''
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter('logs')
#
# with open('test_log.txt') as f:
#     for c in f.readlines():
#         s = c.split(' ')
#         if s[0] == t:
#             n = c.find(':')
#             n1 = c.find('=')
#             n2 = c.rfind('\n')
#             loss = float(c[n + 1:n + 8])
#             epoch = c[n1 + 2:n2 - 1]
#             writer.add_scalar(t, loss, epoch)
# writer.close()
