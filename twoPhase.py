from __future__ import print_function
import sys
# import densenet
import numpy as np
import resnet
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
from torch.autograd import grad
import time
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from copy import copy
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


#### this code is for two-phase warm start experiments ####

parser = argparse.ArgumentParser()
parser.add_argument('--lr1', default=0.001, type=float, help='learning rate for phase 1')
parser.add_argument('--varScale', default=1, type=float, help='variance scale for random merge')
parser.add_argument('--lr2range', default=0, type=int, help='learning magnitude variation for phase 2')
parser.add_argument('--epochs1', default=350, type=int, help='epochs')
parser.add_argument('--epochs2', default=350, type=int, help='epochs')
parser.add_argument('--nWorkers', default=3, type=int, help='epochs')
parser.add_argument('--batchSize1', default=128, type=int, help='batches')
parser.add_argument('--batchSize2', default=128, type=int, help='batches')
parser.add_argument('--decay1', default=0., type=float, help='weight decay 1')
parser.add_argument('--decay2', default=0., type=float, help='weight decay 2')
parser.add_argument('--paramNoise', default=0., type=float, help='std of normal noise to add to parameters after warm start')
parser.add_argument('--split', default=0.5, type=float, help='training phase split')
parser.add_argument('--lr2', default=0.001, type=float, help='learning rate for phase 2')
parser.add_argument('--data', default='CIFAR', type=str, help='svhn or cifar')
parser.add_argument('--model', default='rn', type=str, help='rn, lr, or mlp')
parser.add_argument('--opt', default='sgd', type=str, help='sgd or adam')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--untilGood',  action='store_true', help='train only until train accuracy is 99%')
parser.add_argument('--logGrad', action='store_true', help='log gradients')
parser.add_argument('--logSparse', action='store_true', help='log sparsity')
parser.add_argument('--lastLayerRetrain', action='store_true', help='only retrain last layer (only works for resNet)')
parser.add_argument('--retainAdamParams', action='store_true', help='retain adam parameters in second round?')
parser.add_argument('--lastLayerRetrain2', action='store_true', help='only retrain last layer then retrain whole network')
parser.add_argument('--aug', action='store_true', help='augment training data')
parser.add_argument('--lastOnly', action='store_true', help='train only a second model after the first is fully converged (rather than checkpoints), use this flag to reproduce e.g. table 2 ')
parser.add_argument('--val', action='store_true', help='use validation set instead of test set (1/3 of training data)')
parser.add_argument('--fromBest', action='store_true', help='include training from best')
parser.add_argument('--fname', default='cpt', type=str, help='file name')
args = parser.parse_args()



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.untilGood: args.epochs1 = args.epochs2 = 9999999
if args.lastLayerRetrain2: args.lastLayerRetrain = True

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

nClasses = 10
if args.data == 'CIFAR':
    if args.aug:
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    else:
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
if args.data == 'CIFAR100':
    if args.aug:
        trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)
    else:
        trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transform_test)
        nClasses = int(np.max(trainset.train_labels) + 1)
if args.data == 'SVHN':
    trainset = torchvision.datasets.SVHN(root='data', split='train', download=True, transform=transform_test)
    testset = torchvision.datasets.SVHN(root='data', split='test', download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.nWorkers)

class MyDataset(Dataset):
    def __init__(self):
        self.cifar10 = trainset
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index
    def __len__(self):
        return len(self.cifar10)

dataset = MyDataset()

subset = np.random.permutation([i for i in range(len(trainset))])[:int(args.split * len(trainset))]
trainLoaderFull = DataLoader(dataset, batch_size=args.batchSize2, shuffle=True, num_workers=5)
trainLoaderHalf = DataLoader(dataset, batch_size=args.batchSize1, sampler=SubsetRandomSampler(subset), shuffle=False, num_workers=5)
if args.val:
    subset = np.random.permutation([i for i in range(len(trainset))])
    subTrain = subset[:int(len(trainset) * 2/3.)]
    subTrain1 = subTrain[: int(len(subTrain) * args.split)]
    subVal    = subset[int(len(trainset) * 2/3.) :]
    trainLoaderHalf = DataLoader(dataset, batch_size=args.batchSize1, sampler=SubsetRandomSampler(subTrain1), shuffle=False, num_workers=args.nWorkers)
    trainLoaderFull = DataLoader(dataset, batch_size=args.batchSize2, sampler=SubsetRandomSampler(subTrain), shuffle=False, num_workers=args.nWorkers)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(subVal), shuffle=False, num_workers=args.nWorkers)


class logisticRegression(nn.Module):
    def __init__(self, nc=3, sz=32):
        super(logisticRegression, self).__init__()
        self.nc = nc
        self.sz = sz
        self.lm1 = nn.Linear(nc * sz * sz, nClasses)
    def forward(self, x):
        x = x.view(-1, self.nc * self.sz ** 2)
        return self.lm1(x)


class mlp(nn.Module):
    def __init__(self, nc=3, sz=32):
        super(mlp, self).__init__()
        self.nc = nc
        self.sz = sz
        self.lm1 = nn.Linear(nc * sz * sz, 100)
        self.lm2 = nn.Linear(100, 100)
        self.lm3 = nn.Linear(100, nClasses)
    def forward(self, x):
        x = x.view(-1, self.nc * self.sz ** 2)
        return self.lm3(F.tanh(self.lm2(F.tanh(self.lm1(x)))))

# Model
if args.model == 'rn': net = resnet.ResNet18(num_classes=nClasses).cuda()
if args.model == 'lr': net = logisticRegression().cuda()
if args.model == 'mlp': net = mlp().cuda()
if args.model == '18': net = resnet.ResNet18(num_classes=nClasses).cuda()
if args.model == '34': net = resnet.ResNet34(num_classes=nClasses).cuda()
if args.model == '50': net = resnet.ResNet50(num_classes=nClasses).cuda()
if args.model == '101':net = resnet.ResNet101(num_classes=nClasses).cuda()
if args.model == '152':net = resnet.ResNet152(num_classes=nClasses).cuda()


def printer(print_str):
    sys.stdout.write(print_str + '\n')
    sys.stdout.flush()

printer(str(args.__dict__))

if args.resume:
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr1, momentum=0, weight_decay=args.decay1)
if args.opt == 'adam': optimizer = optim.Adam(net.parameters(), lr=args.lr1, weight_decay=args.decay1)
if args.opt == 'rmsprop': optimizer = optim.RMSprop(net.parameters(), lr=args.lr1, weight_decay=args.decay1)
if args.opt == 'adagrad': optimizer = optim.Adagrad(net.parameters(), lr=args.lr1, weight_decay=args.decay1)


# Training
def train(epoch, net, loader):
    net.train()
    train_loss = 0
    total_loss = 0.
    correct = 0
    total = 0
    count = 0
    lossBig = lossSmall = totalAcc = totalSamps = 0.    
    l2GradSmall = l2GradBig = 0.
    for batch_idx, (inputs, targets, idx) in enumerate(loader):
        count += 1
        optimizer.zero_grad()

        # break up indexes
        ids = idx.numpy(); inBig = []
        for i in ids: inBig.append(i not in subset)
        inBig = np.asarray(inBig)
        indsSmall = np.where(~inBig)[0]
        indsBig = np.where(inBig)[0] 

        if args.logGrad:
            net.eval()
            inp = inputs.numpy()
            tnp = targets.numpy()

            # update from small
            if len(indsSmall) > 0:
                inSmall = Variable(torch.Tensor(inp[indsSmall]).cuda())
                targSmall = Variable(torch.Tensor(tnp[indsSmall]).cuda()).long()
                outputs = net(inSmall)
                loss = criterion(outputs, targSmall)
                loss.backward()
                lossSmall += loss.data.item()
                gradTmp = 0.
                for p in filter(lambda p: p.grad is not None, net.parameters()):
                    gradTmp += torch.sum(p.grad ** 2).data.item()
                gradTmp = np.sqrt(gradTmp)
                l2GradSmall += gradTmp
                net.zero_grad()

            # update from big
            if len(indsBig) > 0:
                inBig = Variable(torch.Tensor(inp[indsBig]).cuda())
                targBig = Variable(torch.Tensor(tnp[indsBig]).cuda()).long()
                outputs= net(inBig)
                loss = criterion(outputs, targBig) 
                loss.backward()
                lossBig += loss.data.item()
                gradTmp = 0.
                for p in filter(lambda p: p.grad is not None, net.parameters()):
                    gradTmp += torch.sum(p.grad ** 2).data.item()
                gradTmp = np.sqrt(gradTmp) 
                l2GradBig += gradTmp
                net.zero_grad()
            net.train()

        # standard update
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        total_loss += loss.data.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().data.item()
        totalAcc += correct
        totalSamps += total
        if batch_idx % 10 == 0 and batch_idx != 0:
            printer(str(epoch) + 
                    '\t' + str(batch_idx) + 
                    '\t' + str(train_loss/10.) + 
                    '\t' + str(100.*correct/total) + 
                    '\t' + str(l2GradSmall / 10.) + 
                    '\t' + str(l2GradBig / 10.) + 
                    '\t' + str(lossSmall / 10.) + 
                    '\t' + str(lossBig / 10.)) 
            correct = total = train_loss = l2GradSmall = l2GradBig = lossSmall = lossBig = 0.
    l2norm = 0.0
    for p in filter(lambda p: p.grad is not None, net.parameters()):
        l2norm += p.norm(2).item()
    print('l2:\t' + str(l2norm)) 
    return (totalAcc / totalSamps), (total_loss / count)

def getl2(net):
    l2norm = 0.0
    for p in net.parameters():
        l2norm += p.norm(2).item()
    print('init l2:\t' + str(l2norm))

def test(epoch, net):
    global best_acc
    global best_net
    global best_iter
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += (loss.data.item() * targets.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().data.item() 
        printer('Test:\t' + str(epoch) + '\t' + str(test_loss/total) +'\t' + str(100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_iter = epoch
        best_acc = acc
        net = net.cpu()
        best_net = deepcopy(net)
        net = net.cuda()


if args.model == 'lr': netsAll = [logisticRegression().cpu()]
if args.model == 'mlp':netsAll = [mlp().cpu()]
if args.model == 'rn': netsAll = [resnet.ResNet18().cpu()]
if args.model == '18': netsAll = [resnet.ResNet18(num_classes=nClasses).cpu()]
if args.model == '34': netsAll = [resnet.ResNet34(num_classes=nClasses).cpu()]
if args.model == '50': netsAll = [resnet.ResNet50(num_classes=nClasses).cpu()]
if args.model == '101':netsAll = [resnet.ResNet101(num_classes=nClasses).cpu()]
if args.model == '152':netsAll = [resnet.ResNet152(num_classes=nClasses).cpu()]

# initial train
saveEvery = 5

if args.split == 1: getl2(net)
for epoch in range(args.epochs1):
    start = time.time()
    trainAcc, trainLoss = train(epoch, net, trainLoaderHalf)
    end = time.time()
    printer('Train:\t' + str(epoch) + '\t' + str(trainLoss) + '\t' + str(trainAcc * 100) + '\t' + str(end-start))
    test(epoch, net)
    if args.untilGood:
        if trainAcc > .99:
           break
    if ((epoch + 1) % saveEvery == 0): 
        net = net.cpu()
        netsAll.append(deepcopy(net))
        net = net.cuda()

net = net.cpu()
if args.split == 1: sys.exit()

# add parameter noise
if args.paramNoise > 0:
    for p in filter(lambda p: p.grad is not None, net.parameters()):
        p.data += torch.Tensor(np.random.randn(*p.size()) * np.sqrt(args.paramNoise))

# retrain from converged model
if args.lastOnly:
    bestnet = deepcopy(best_net)
    lastnet = deepcopy(net)
    if args.fromBest:
        printer('training from best network, ' + str(best_iter))
        bestnet = bestnet.cuda()
        paramVec = betnet.parameters()
        if args.lastLayerRetrain: paramVec = bestnet.module.linear.parameters()
        if args.opt == 'sgd': optimizer = optim.SGD(paramVec, lr=args.lr2, momentum=0, weight_decay=args.decay2)
        if args.opt == 'adam': optimizer = optim.Adam(paramVec, lr=args.lr2, weight_decay=args.decay2)
        if args.opt == 'rmsprop': optimizer = optim.RMSprop(paramVec, lr=args.lr2, weight_decay=args.decay2)
        if args.opt == 'adagrad': optimizer = optim.Adagrad(paramVec, lr=args.lr2, weight_decay=args.decay2)
        for epoch in range(args.epochs2):
           start = time.time()
           trainAcc, trainLoss = train(epoch, bestnet, trainLoaderFull)
           end = time.time()
           printer('Train:\t' + str(epoch) + '\t' + str(trainLoss) + '\t' + str(trainAcc * 100) + '\t' + str(end-start))
           test(epoch, bestnet)
           if args.untilGood:
               if trainAcc > .99: 
                   break
        bestnet = bestnet.cpu()

    lr2s = [args.lr2]
    const = 10
    for i in range(args.lr2range):
        lr2s.append(args.lr2 * const)
        lr2s.append(args.lr2 / const)
        const *= 10

    # if given multiple second learning rates   
    for lr in lr2s:
        printer('training from last network ' + str(lr))
        ln = deepcopy(lastnet)
        ln = ln.cuda()
        getl2(ln)
        paramVec = ln.parameters()
        if args.lastLayerRetrain: paramVec = ln.module.linear.parameters()
        if args.opt == 'sgd': optimizer = optim.SGD(paramVec, lr=lr, momentum=0, weight_decay=args.decay2)
        if args.opt == 'adam': optimizer = optim.Adam(paramVec, lr=lr, weight_decay=args.decay2)
        if args.opt == 'rmsprop': optimizer = optim.RMSprop(paramVec, lr=lr, weight_decay=args.decay2)
        if args.opt == 'adagrad': optimizer = optim.Adagrad(paramVec, lr=lr, weight_decay=args.decay2)
        for epoch in range(args.epochs2):
           start = time.time()
           trainAcc, trainLoss = train(epoch, ln, trainLoaderFull)
           end = time.time()
           printer('Train:\t' + str(epoch) + '\t' + str(trainLoss) + '\t' + str(trainAcc * 100) + '\t' + str(end-start))
           test(epoch, ln)
           if args.untilGood:
               if trainAcc > .99:
                   break
        ln = ln.cpu()
        net = net.cpu()

        pVec1 = np.asarray([])
        for p in net.parameters(): pVec1 = np.concatenate((pVec1, p.data.numpy().flatten()))
        pVec2 = np.asarray([])
        for p in ln.parameters(): pVec2 = np.concatenate((pVec2, p.data.numpy().flatten()))

        from scipy.stats.stats import pearsonr
        from scipy.stats import spearmanr
        from scipy.spatial.distance import cosine
        printer('Correlation pear:\t' + str(pearsonr(pVec1, pVec2)[0])) 
        printer('Correlation spear:\t' + str(spearmanr(pVec1, pVec2).correlation)) 
        printer('Correlation dist:\t' + str(np.sqrt(sum((pVec1 - pVec2) ** 2)))) 
        printer('Correlation dist cos:\t' + str(cosine(pVec1, pVec2))) 
else:
    for i in range(len(netsAll)):
        net = deepcopy(netsAll[i])
        net = net.cuda()

        paramVec = net.parameters()
        if args.lastLayerRetrain: paramVec = net.module.linear.parameters()
        if args.opt == 'sgd': optimizer = optim.SGD(paramVec, lr=args.lr2, momentum=0, weight_decay=args.decay2)
        if args.opt == 'adam': optimizer = optim.Adam(paramVec, lr=args.lr2, weight_decay=args.decay2)
        if args.opt == 'rmsprop': optimizer = optim.RMSprop(paramVec, lr=args.lr2, weight_decay=args.decay2)
        if args.opt == 'adagrad': optimizer = optim.Adagrad(paramVec, lr=args.lr2, weight_decay=args.decay2)
        printer('training from epoch ' + str(i * saveEvery))
        for epoch in range(args.epochs2):
            start = time.time()
            trainAcc, trainLoss = train(epoch, net, trainLoaderFull)
            end = time.time()
            printer('Train:\t' + str(epoch) + '\t' + str(trainLoss) + '\t' + str(trainAcc * 100) + '\t' + str(end-start))
            test(epoch, net)
            if args.untilGood:
                if trainAcc > .99:
                    break
    net = net.cpu(
)

# do a second round of retraining after retraining last layer
if args.lastLayerRetrain2:
    ln = deepcopy(ln).cuda()
    paramVec = ln.parameters()
    printer('doing second warm start')
    if args.opt == 'sgd': optimizer = optim.SGD(paramVec, lr=args.lr2, momentum=0, weight_decay=args.decay2)
    if args.opt == 'adam': optimizer = optim.Adam(paramVec, lr=args.lr, weight_decay=args.decay2)
    if args.opt == 'rmsprop': optimizer = optim.RMSprop(paramVec, lr=args.lr, weight_decay=args.decay2)
    if args.opt == 'adagrad': optimizer = optim.Adagrad(paramVec, lr=args.lr, weight_decay=args.decay2)
    for epoch in range(args.epochs2):
       start = time.time()
       trainAcc, trainLoss = train(epoch, ln, trainLoaderFull)
       end = time.time()
       printer('Train:\t' + str(epoch) + '\t' + str(trainLoss) + '\t' + str(trainAcc * 100) + '\t' + str(end-start))
       test(epoch, ln)
       if args.untilGood:
           if trainAcc > .99 and epoch >= 349:
               break




