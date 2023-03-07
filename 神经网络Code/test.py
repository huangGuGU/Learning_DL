import os
import torch
import  torchvision
from PIL import Image

right_num = 0
false_img = []
img_path = 'test_img'
use_model = 'logs/Hao9.pth'
img_list = os.listdir(img_path)
for img_name in img_list:
    Img_path = os.path.join(img_path,img_name)
    with open(Img_path) as f:
        img = Image.open(Img_path)
        transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize(28),
                                                    torchvision.transforms.Grayscale(),
                                                    torchvision.transforms.ToTensor()
                                                    ])
        img_reverse = 1 - transform(img)

        model = torch.load(use_model)
        IMG = torch.reshape(img_reverse, (1, 1, 28, 28))

        model.eval()
        with torch.no_grad():
            output = model(IMG)
        print('预测值：',int(output.argmax(1)))
        print('实际值：',img_name[:-4])

        if int(output.argmax(1)) == int(img_name[-5:-4]):
            print('right')
            right_num+=1

        else:
            print('false')
            false_img.append(img_name)


        print('-'*20)


print('''
使用的权重：{}
正确的个数：{}
图片个数：{}
正确率：{}
错误的图片：{}
'''.format(use_model,right_num,len(img_list),right_num/len(img_list),false_img))