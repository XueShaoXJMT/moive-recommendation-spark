import argparse
import os.path as osp
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fish import fish
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=20)  # 最大迭代次数
    parser.add_argument('--save-epoch', type=int, default=5)  # 保存周期
    parser.add_argument('--shot', type=int, default=5)  # 各类别支撑集样本数
    parser.add_argument('--query', type=int, default=15)  # 各类别验证集样本数
    parser.add_argument('--train-way', type=int, default=15)  # 训练时单次采样的类别数
    parser.add_argument('--test-way', type=int, default=5)  # 验证时单次采样的类别数
    parser.add_argument('--save-path', default='./save/proto-5-2')  # 模型字典的保存路径
    parser.add_argument('--load', default='./save/proto-5/max-acc.pth')  # 模型字典的加载路径
    parser.add_argument('--gpu', default='0')  # 使用GPU
    args = parser.parse_args()
    pprint(vars(args))
    warnings.filterwarnings('ignore')
    set_gpu(args.gpu)
    ensure_path(args.save_path)

    #trainset = MiniImageNet('train')
    #trainset = fish('fish100_train')
    trainset = fish('train50')
    train_sampler = CategoriesSampler(trainset.labels, 20,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=0, pin_memory=True)#num_workers=8

    valset = fish('val30')
    val_sampler = CategoriesSampler(valset.labels, 20,
                                    args.test_way, args.shot + args.query)#400
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=True)#num_workers=8

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))#从上次训练的模型参数，继续训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item()))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()


        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            pred = torch.argmax(logits, dim=1)
            #precision, recall, F1, _ = precision_recall_fscore_support(label.cpu().detach(), pred.cpu().detach(),
                                                                       #average="macro")
            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f}'.format(epoch, vl))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

