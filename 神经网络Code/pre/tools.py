# import os
#
# import numpy as np
# from PIL import Image
#
# path = '2'
# target = 'y'
# img_list = os.listdir(path)
# for i in img_list:
#     img_path = os.path.join(path,i)
#     target_path = os.path.join(target,i)
#     image = Image.open(img_path)
#     img = 255-np.array(image)
#     img = Image.fromarray(img)
#     img.save(target_path)
# print('done')


import torch
import torch.nn as nn


# class MyModule(nn.Module):
#     def __init__(self, num):
#         super(MyModule, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = nn.Parameter(params)
#
#     def forward(self, x):
#         y = self.params * x
#
#
# my_module = MyModule(10)
# inputs = torch.ones(10)
# outputs = my_module(inputs)
# print(my_module.state_dict())
# print(list(my_module.parameters()))
# print(dict(my_module.named_parameters()))

######################################################################################################################################################################


def Phase_max_pooling(feature_map, size, stride):

        b,c,h,w = feature_map.shape

        out_height = (h - size) // stride + 1
        out_width = (w - size) // stride + 1
        out_pooling = torch.zeros((b,c,out_height, out_width), dtype=torch.float32)
        i = j = 0
        for m in range(0, h, stride):
            for n in range(0, w, stride):
                 if (n + stride) <= w and (m + stride) <= h:
                    a = torch.max(torch.angle(feature_map[:,:,m: m + size, n: n + size]).reshape(b,c,-1),-1)

                    out_pooling[:, :, i, j] =a.values
                    j += 1
            i += 1
            j = 0
        return out_pooling



input = torch.complex(torch.rand(size = (3,1,8,8)),torch.rand(size = (3,1,8,8)))



# c=[]
# a = torch.rand(size = (3,1,1,1))
# input = torch.rand(size = (3,1,4,4))
# print('input',input)
# for i in range(input.shape[0]):
a = Phase_max_pooling(input,2,2)

print(a)





######################################################################################################################################################################


# def nearest(image, target_size):
#     """
#     Nearest Neighbour interpolate for RGB  image
#
#     :param image: rgb image
#     :param target_size: tuple = (height, width)
#     :return: None
#     """
#     if target_size[0] < image.shape[0] or target_size[1] < image.shape[1]:
#         raise ValueError("target image must bigger than input image")
#     # 1：按照尺寸创建目标图像
#     target_image = torch.zeros(size=(4,4))
#     # 2:计算height和width的缩放因子
#     alpha_h = target_size[0] / image.shape[0]
#     alpha_w = target_size[1] / image.shape[1]
#
#     for tar_x in range(target_image.shape[0] - 1):
#         for tar_y in range(target_image.shape[1] - 1):
#             # 3:计算目标图像人任一像素点
#             # target_image[tar_x,tar_y]需要从原始图像
#             # 的哪个确定的像素点image[src_x, xrc_y]取值
#             # 也就是计算坐标的映射关系
#             src_x = round(tar_x / alpha_h)
#             src_y = round(tar_y / alpha_w)
#
#             # 4：对目标图像的任一像素点赋值
#             target_image[tar_x, tar_y] = image[src_x, src_y]
#
#     return target_image






# b = torch.zeros(size = (3,1,4,4))
# a = torch.rand(size = (3,1,2,2))
# print(a)
# for i in range(a.shape[0]):
#     b[i][0] = nearest_(a[i][0], (4,4))
#
# print(b)