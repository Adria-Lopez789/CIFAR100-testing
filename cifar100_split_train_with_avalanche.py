'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from avalanche.benchmarks.classic import SplitCIFAR100

import sys
import os
import argparse

from models import *
import numpy as np
from copy import deepcopy

import logging

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--chkpt', default=0, type=int, help='checkpoint number')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum','-m', default=0.9, type=float, help='momentum')
args = parser.parse_args()
chkpt_no = args.chkpt
logging.basicConfig(filename="logs/logs_"+str(chkpt_no)+".log", filemode='w', level=logging.DEBUG)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#trainloader_complete = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
trainloader_complete = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
model = ResNet18()
model = model.to(device)
if device == 'cuda':
    #model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Training
def train(epoch, trainloader, run_test=False):
    print('\nEpoch: %d' % epoch)

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_cnt = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        test_train(epoch, testloader, "Before Batch Train ")
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accbatch = predicted.eq(targets).sum().item()

        print('Train Batch Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), accbatch, len(inputs),
                accbatch / len(inputs)))

        test_train(epoch, testloader, "After Batch Train ")
        test_train_av(epoch, split_testloader, "After Split Batch Train ")

    logging.info("Train Epoch %d Loss %f Accuracy %f",epoch,train_loss/(batch_idx+1), correct/total)

def test_train(epoch, trainloader, msg="Test Train "):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

           _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(inputs), len(testloader.dataset),
            #    100. * batch_idx / len(testloader), loss.item()))

    logging.info("Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)
    print(msg+"Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)

def train_av(epoch, trainloader, run_test=False):
    print('\nEpoch: %d' % epoch)

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_cnt = 0
    batch_idx = 0
    #for batch_idx, (inputs, targets) in enumerate(trainloader):
    for (inputs, targets) in enumerate(trainloader):
        inputs, targets = torch.FloatTensor(targets[0]).to(device), torch.LongTensor(np.array(targets[1])).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        #test_train(epoch, testloader, "Before Batch Train ")
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accbatch = predicted.eq(targets).sum().item()

        print('Train Batch Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), accbatch, len(inputs),
                accbatch / len(inputs)))
        batch_idx = batch_idx + 1

    logging.info("Train Epoch %d Loss %f Accuracy %f",epoch,train_loss/(batch_idx+1), correct/total)

def test_train_av(epoch, trainloader, msg="Test Train "):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
   total = 0
    batch_idx = 0
    with torch.no_grad():
        #for batch_idx, (inputs, targets) in enumerate(trainloader):
        for (inputs, targets) in enumerate(trainloader):
            inputs, targets = torch.FloatTensor(targets[0]).to(device), torch.LongTensor(np.array(targets[1])).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_idx = batch_idx + 1

            #print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(inputs), len(testloader.dataset),
            #    100. * batch_idx / len(testloader), loss.item()))

    logging.info("Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)
    print(msg+"Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(inputs), len(testloader.dataset),
            #    100. * batch_idx / len(testloader), loss.item()))

    logging.info("Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)
    print("Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)

# creating the benchmark (scenario object)
split_cifar100 = SplitCIFAR100(
    n_experiences=2,
    seed=1234,
    train_transform = transform_train,
    eval_transform = transform_test,
    #fixed_class_order=list(range(100)),
    #shuffle=False,
)

# recovering the train and test streams
train_stream = split_cifar100.train_stream
test_stream = split_cifar100.test_stream
split_testloader = testloader

cnt = 0
# iterating over the train stream
for experience in train_stream:
    print("Start of task ", experience.task_label)
    print('Classes in this task:', experience.classes_in_this_experience)

    # The current Pytorch training set can be easily recovered through the
    # experience
    current_training_set = experience.dataset
    #split_train_dataset = SplitCIFAR100(n_experiences=2, first_exp_with_half_classes=True, train_transform=transform_train, eval_transform=transform_test, dataset_root='./data')
    split_trainloader = torch.utils.data.DataLoader(current_training_set, batch_size=64, shuffle=True, num_workers=2)
    # ...as well as the task_label
    print('Task {}'.format(experience.task_label))
    print('This task contains', len(current_training_set), 'training examples')

    # we can recover the corresponding test experience in the test stream
    current_test_set = test_stream[experience.current_experience].dataset
    print('This task contains', len(current_test_set), 'test examples')
    split_testloader = torch.utils.data.DataLoader(current_test_set, batch_size=64, shuffle=False, num_workers=2)

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_av(epoch, split_trainloader)
        test_train_av(epoch, split_testloader)
        test(0)

    torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'loss': loss,
                }, "./checkpoint/ckpt_"+str(chkpt_no)+"_"+str(cnt)+".pth")
    cnt = cnt + 1

#for epoch in range(start_epoch, start_epoch+args.epochs):
#    train(epoch,trainloader_complete)
#    test_train_av(epoch, split_testloader)
#    test(epoch)
