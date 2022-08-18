import csv
import os
import os.path as osp
from PIL import Image
import pandas
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import RandomCrop
import codecs

ROOT_PATH = './materials/fish100/'

class fish(data.Dataset):

    def __init__(self, setname):
        #f = codecs.open('./materials/fish100/train.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
        path = osp.join(ROOT_PATH, setname + '.txt')
        f=codecs.open(path,'r','utf-8')
        line = f.readline()  # 以行的形式进行读取文件
        label = []
        name = []
        while line:
            a = line.split()
            b = a[:1]  # 这是选取需要读取的位数
            str1 = ''.join(b)  # 通过join函数，将list转化为str
            path = osp.join(str1)
            #print(path)
            c = a[1:]
            str2 = ''.join(c)
            name.append(path)  # 将其添加在列表之中
            label.append(int(str2))
            line = f.readline()
        f.close()

        self.data = name
        self.labels = label
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((84 * 8 // 7, 84 * 8 // 7)),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.labels[i]
        #print(path)
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
