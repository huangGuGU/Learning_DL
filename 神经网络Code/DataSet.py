
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from PIL import Image





class MyDataset(Dataset):
    def __init__(self, imgFile, transform , transform1,loader):
        images = []
        with open(imgFile, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                images.append((temp[0], temp[1]))
        self.images = images
        self.transform = transform
        self.transform1 = transform1
        self.loader = loader

    def __getitem__(self, index):
        img1, label = self.images[index]
        img1 = self.loader(img1)

        label = self.loader(label)

        img1 = transforms.ToTensor()(img1).float()
        label = transforms.ToTensor()(label).float()
        if self.transform is not None:
            img1 = self.transform(img1)

            label = self.transform1(label)



        return [img1, label]

    def __len__(self):
        return len(self.images)




class ImgLoader:
    def __init__(self):
        self.norm_flag = True
        self.min_max_list = [0,255]

    def img_norm(self, data):
        if self.norm_flag is None:
            return data
        min_max = self.min_max_list
        _range = min_max[1] - min_max[0]
        return (data - min_max[0]) / _range

    def __call__(self, path):
        ret = Image.open(path).convert('L')
        return self.img_norm(np.asarray(ret, dtype=np.float))








