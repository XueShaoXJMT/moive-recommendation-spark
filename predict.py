import json
import os

import torch
from torch.autograd import Variable

from convnet import Convnet

model = Convnet().cuda()
model.load_state_dict(torch.load('./save/proto-5/max-acc.pth'))
model.eval()
#model = get_model1()
image_class_index = json.load(open('D:\pythonproject\predict\imagenet_class_index.json'))
#image_class_index = json.load(open('.\img\image_class_index.json'))

def get_prediction(loader):
    for x, y in loader:
        x = Variable(x.cuda())
        out = model(x)
        #print(out)
        pred = torch.max(out, 1)[1]
        xy = str(pred.item())
    return image_class_index[xy]

def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
