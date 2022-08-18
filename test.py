import argparse
import warnings
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms

from fish import fish
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric, count_rec, count_F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')  # 使用GPU
    parser.add_argument('--load', default='./save/proto-5/max-acc.pth')  # 加载模型字典的路径
    parser.add_argument('--batch', type=int, default=100)  # 批大小
    parser.add_argument('--way', type=int, default=5)  # 单次采样类别数
    parser.add_argument('--shot', type=int, default=5)  # 各类别支撑集样本数
    parser.add_argument('--query', type=int, default=15)  # 各类别验证集样本数
    args = parser.parse_args()
    pprint(vars(args))
    warnings.filterwarnings('ignore')
    set_gpu(args.gpu)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset=fish('test20')
    sampler = CategoriesSampler(dataset.labels,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=0, pin_memory=True)  # num_workers=8

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):
        #print(i)
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape(args.shot, args.way, -1).mean(dim=0)
        p = x

        logits = euclidean_metric(model(data_query), p)
        #print(logits)
        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
        #print(label)
        acc = count_acc(logits, label)
        #rec = count_rec(logits, label)
        #F = count_F(pre,rec)
        pred = torch.argmax(logits, dim=1)
        precision, recall, F, _ = precision_recall_fscore_support(label.cpu().detach(), pred.cpu().detach(), average="macro")
        #ave_acc.add(acc)
        #print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        #print('batch {}: pre: {:.2f}'.format(i, pre))
        print('batch {}: pre={:.4f}, rec={:.4f}, F={:.4f}'.format(i, precision, recall, F))
        x = None;
        p = None;
        logits = None

