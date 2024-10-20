'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from avalanche.benchmarks.classic import SplitCIFAR100
import wandb

import sys
import os
import argparse

from resnet import *
import numpy as np
from copy import deepcopy

import logging

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--chkpt', default=3, type=int, help='checkpoint number')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum','-m', default=0.9, type=float, help='momentum')
parser.add_argument('--print-frequency', default=2, type=int)
parser.add_argument('--start-from-checkpoint', default="", type=str)
parser.add_argument('--continual-type', default="CL", type=str)
#parser.add_argument('--start-from-checkpoint', default="", type=str)

args = parser.parse_args()
chkpt_no = args.chkpt
logging.basicConfig(filename="logs/logs_"+str(chkpt_no)+".log", filemode='w', level=logging.DEBUG)



data = torch.tensor([[1.0, 2.0, 3.0],
                     [2.0, 4.0, 6.0]])

# Compute the covariance matrix
cov_matrix = torch.cov(data)

print(cov_matrix)
wandb.login(key="6f35bcdd0108305865b662cb61e4572421e36d7a")
wandb.init(
    project="cifar100_50-100_CL",
    name="Control set (50 new classes)",
    config= {"lr":0.01,
             "batch-size":64,
             "momentum":0.9,
             "epochs":"100x2"
            }
)



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

# Model
print('==> Building model..')
model = ResNet18()
model = model.to(device)
if args.start_from_checkpoint != "":
    checkpoint = torch.load(args.start_from_checkpoint, map_location=device)

    model_state_dict = checkpoint['model_state_dict']

    model.load_state_dict(model_state_dict)
if device == 'cuda':
    #model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)



def train_av(epoch, trainloader, run_test=False):
    print('\nEpoch: %d' % epoch)

    model.train()
    cumulative_loss_before_batch = 0
    cumulative_loss_after_batch = 0
    correct_before_batch = 0
    correct_after_batch = 0
    total_data_samples = 0
    batch_cnt = 1
    batch_idx = 0
    #for batch_idx, (inputs, targets) in enumerate(trainloader):
    for (inputs, targets) in enumerate(trainloader):
        inputs, targets = torch.FloatTensor(targets[0]).to(device), torch.LongTensor(np.array(targets[1])).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_before_batch = criterion(outputs, targets)
        cumulative_loss_before_batch += loss_before_batch

        #Before we do the batch update step, we track the train accuracy
        _, predicted_before_batch = outputs.max(1)
        total_data_samples += targets.size(0)
        correct_before_batch += predicted_before_batch.eq(targets).sum().item()
        wandb.log({"train_loss_before_batch_update": cumulative_loss_before_batch / (batch_idx+1),
                   "train_accuracy_before_batch_update": 100. * correct_before_batch / total_data_samples,
                   "epoch:epoch": epoch
                   })
        loss_before_batch.backward()
        optimizer.step()

        outputs = model(inputs)
        loss_after_batch = criterion(outputs, targets)
        _, predicted_after_batch = outputs.max(1)
        correct_after_batch += predicted_after_batch.eq(targets).sum().item()
        cumulative_loss_after_batch += loss_after_batch.item()
        batch_accuracy = predicted_after_batch.eq(targets).sum().item()

        wandb.log({"train_loss_after_batch_update": cumulative_loss_after_batch / (batch_idx+1),
                   "train_accuracy_after_batch_update": 100. * correct_after_batch / total_data_samples,
                   "epoch:epoch": epoch
                   })
        if batch_idx % args.print_frequency == 0:
            print('Train Batch Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss_after_batch.item(), batch_accuracy, len(inputs),
                    batch_accuracy / len(inputs)))
        batch_idx = batch_idx + 1

    logging.info("Train Epoch %d Loss %f Accuracy %f",epoch,cumulative_loss_after_batch/(batch_idx+1), correct_after_batch/total_data_samples)

def test_train_av(epoch, testloader, msg="Test Train "):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    with torch.no_grad():

        for (inputs, targets) in enumerate(testloader):
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

        wandb.log({
            "test_loss": test_loss / (batch_idx+1),  # Divide by batch_idx for average loss
            "test_accuracy": 100. * correct / total,  # Divide by total number of samples for accuracy
            "epoch": epoch
        })
    logging.info("Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)
    print(msg+"Test Epoch %d Loss %f Accuracy %f", epoch, test_loss/(batch_idx+1), correct/total)

if(args.continual_type == "CI"):
    # creating the benchmark (scenario object)
    split_cifar100 = SplitCIFAR100(
        n_experiences=2,
        seed=1234,
        train_transform = transform_train,
        eval_transform = transform_test,
        first_exp_with_half_classes=False,
        #fixed_class_order=list(range(100)),
        #shuffle=False,
    )

        # recovering the train and test streams
    train_stream = split_cifar100.train_stream
    test_stream = split_cifar100.test_stream

    first_train_stream_dataset = train_stream[0].dataset
    first_test_stream_dataset = test_stream[0].dataset

    second_train_stream_dataset = train_stream[0].dataset + train_stream[1].dataset
    second_test_stream_dataset = test_stream[0].dataset + test_stream[1].dataset

elif(args.continual_type == "CL" ):
    split_cifar100 = SplitCIFAR100(
        n_experiences=1,
        seed=1234,
        train_transform = transform_train,
        eval_transform = transform_test,
        first_exp_with_half_classes=False,
        #fixed_class_order=list(range(100)),
        #shuffle=False,
    )
    train_stream_dataset = split_cifar100.train_stream[0].dataset
    test_stream_dataset = split_cifar100.test_stream[0].dataset

    total_size = len(train_stream_dataset)
    split_sizes = [int(total_size * 0.5), total_size - int(total_size * 0.5)]  # Splitting into two equal parts

    # Create the two subsets
    first_train_dataset, subset2 = torch.utils.data.random_split(train_stream_dataset, split_sizes)
    second_train_dataset = first_train_dataset + subset2

    first_train_stream_dataset = train_stream[0].dataset
    first_test_stream_dataset = test_stream[0].dataset

    second_train_stream_dataset = train_stream[0].dataset + train_stream[1].dataset
    second_test_stream_dataset = test_stream[0].dataset + test_stream[1].dataset



#split_train_dataset = SplitCIFAR100(n_experiences=2, first_exp_with_half_classes=True, train_transform=transform_train, eval_transform=transform_test, dataset_root='./data')
first_split_trainloader = torch.utils.data.DataLoader(first_train_stream_dataset, batch_size=64, shuffle=True, num_workers=2)
second_split_trainloader = torch.utils.data.DataLoader(second_train_stream_dataset, batch_size=64, shuffle=True, num_workers=2)


# we can recover the corresponding test experience in the test stream
first_split_testloader = torch.utils.data.DataLoader(first_test_stream_dataset, batch_size=64, shuffle=False, num_workers=2)
second_split_testloader = torch.utils.data.DataLoader(second_test_stream_dataset, batch_size=64, shuffle=False, num_workers=2)

#We do a first epoch with the trained path, to see its accuracy
if(args.start_from_checkpoint != ""):
    epoch = 0
    train_av(epoch, first_split_trainloader)
    test_train_av(epoch, first_split_testloader)

else:
    #We align both graphs
    for (inputs, targets) in enumerate(first_split_trainloader):
        wandb.log({"train_loss_after_batch_update": 0,
                   "train_accuracy_after_batch_update": 0,
                   "epoch:epoch": 0
                   })
        wandb.log({"train_loss_before_batch_update": 0,
                   "train_accuracy_before_batch_update": 0,
                   "epoch:epoch": 0
                   })
    wandb.log({"test_loss": 0,
               "test_accuracy": 0,
               "epoch:epoch": 0
               })
for epoch in range(start_epoch, start_epoch+args.epochs):
    train_av(epoch, second_split_trainloader)
    test_train_av(epoch, second_split_testloader)

torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss,
            }, "./checkpoint/ckpt_"+str(chkpt_no)+"_1.pth")


#for epoch in range(start_epoch, start_epoch+args.epochs):
#    train(epoch,trainloader_complete)
#    test_train_av(epoch, split_testloader)
#    test(epoch)
