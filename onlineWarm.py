from __future__ import print_function
import sys
# import densenet
import numpy as np
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
import time
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from torchvision import datasets, transforms
import resnet
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA 

### code for online learning experiments with the shrink-perturb trick ###

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for phase 1')
parser.add_argument('--noiseScale', default=1, type=float, help='noise scale')
parser.add_argument('--nQuery', default=1000, type=int, help='number of query samps')
parser.add_argument('--nWorkers', default=3, type=int, help='epochs')
parser.add_argument('--batchSize1', default=128, type=int, help='batches')
parser.add_argument('--batchSize2', default=128, type=int, help='batches')
parser.add_argument('--decay', default=0., type=float, help='weight decay')
parser.add_argument('--goodNum', default=0.99, type=float, help='at what training accuracy to stop learning?')
parser.add_argument('--lambd', default=0, type=float, help='amount random')
parser.add_argument('--model', default='rn', type=str, help='model type')
parser.add_argument('--data', default='cifar', type=str, help='model type')
args = parser.parse_args()



# Data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.data == 'cifar':
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
if args.data == 'svhn':
    trainset = torchvision.datasets.SVHN(root='data', split='train', download=True, transform=transform_test)
    testset = torchvision.datasets.SVHN(root='data', split='test', download=True, transform=transform_test)

class MyDataset(Dataset):
    def __init__(self, trainset):
        self.cifar10 = trainset 
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

dataset = MyDataset(trainset)



subset = np.random.permutation([i for i in range(len(trainset))])[:len(trainset)]
trainLoaderFull = DataLoader(dataset, batch_size=args.batchSize2, shuffle=True, num_workers=5)
trainLoaderHalf = DataLoader(dataset, batch_size=args.batchSize1, sampler=SubsetRandomSampler(subset), shuffle=False, num_workers=5)

subTrain = subset[:int(len(trainset) * 2/3.)]
subVal   = subset[int(len(trainset) * 2/3.):]
testloader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(subVal), shuffle=False, num_workers=args.nWorkers)


class mlp(nn.Module):
    def __init__(self, nc=3, sz=32):
        super(mlp, self).__init__()
        self.nc = nc
        self.sz = sz
        self.lm1 = nn.Linear(nc * sz * sz, 100)
        self.lm2 = nn.Linear(100, 100)
        self.lm3 = nn.Linear(100, 10)
    def forward(self, x):
        x = x.view(-1, self.nc * self.sz ** 2)
        return self.lm3(F.relu(self.lm2(F.relu(self.lm1(x)))))


# Model
# net = VGG('VGG19')
if args.model == 'rn':
    net = resnet.ResNet18().cuda()
if args.model == 'mlp':
    net = mlp().cuda()


def printer(print_str):
    sys.stdout.write(print_str + '\n')
    sys.stdout.flush()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

printer(str(args.__dict__))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0, weight_decay=args.decay)
optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=args.decay)

# Training
def train(epoch, net, loader):
    net.train()
    train_loss = 0
    total_loss = 0.
    totalAcc = totalSamps = count = total = correct = 0
    for batch_idx, (inputs, targets, idx) in enumerate(loader):
        count += 1
        optimizer.zero_grad()

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
            correct = total = train_loss = 0.
    return (totalAcc / totalSamps), (total_loss / count)


from copy import copy
def shrink(net, lambd):
    if args.model == 'rn':
        newNet = resnet.ResNet18().cuda()
    if args.model == 'mlp':
        newNet = mlp().cuda()
    params1 = newNet.parameters()
    params2 = net.parameters()
    for p1, p2 in zip(*[params1, params2]):
        p1.data = deepcopy(args.noiseScale * p1.data + lambd * p2.data)
    return newNet


def test(epoch, net):
    global best_acc
    global best_net
    global best_iter
    test_loss = 0
    correct = 0
    total = 0
    
    net = net.cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.cuda()
            outputs = net(inputs.cuda())
            loss = criterion(outputs, targets)
            test_loss += (loss.data.item() * targets.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().data.item()
        printer('Test:\t' + str(epoch) + '\t' + str(test_loss/total) +'\t' + str(100.*correct/total))

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()



nTrain = args.nQuery
currentSubset = np.asarray([])
trainLoaderAll = DataLoader(dataset, batch_size=args.batchSize1, shuffle=False, num_workers=5)

while len(currentSubset) != len(subTrain):
    selectedInds = subTrain[nTrain-args.nQuery:nTrain]
    currentSubset = np.concatenate((selectedInds, currentSubset)).astype(int)
    dataset = MyDataset(trainset)
    trainLoaderHalf = DataLoader(dataset, batch_size=args.batchSize1, sampler=SubsetRandomSampler(currentSubset), shuffle=False, num_workers=5)
    epoch = 0
    net = shrink(net, args.lambd)
    net = net.train()
    optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=args.decay)

    # train until convergence
    while True:
        start = time.time()
        trainAcc, trainLoss = train(epoch, net, trainLoaderHalf)
        end = time.time()
        printer('Train ' + str(len(currentSubset)) +  ':\t' + str(epoch) + '\t' + str(trainLoss) + '\t' + str(trainAcc * 100) + '\t' + str(end-start))
        if trainAcc >= args.goodNum:
           break
        epoch += 1
    net = net.eval()
    test(len(currentSubset), net)
    nTrain += args.nQuery

