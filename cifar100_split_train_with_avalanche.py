'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr-start', default=0.00001, type=float, help='learning rate')
parser.add_argument('--lr-increase', default=False, type=bool, help='learning rate')
parser.add_argument('--momentum','-m', default=0.9, type=float, help='momentum')
parser.add_argument('--print-frequency', default=2, type=int)
parser.add_argument('--start-from-checkpoint', default="", type=str)
parser.add_argument('--continual-type', default="CL", type=str)
#parser.add_argument('--start-from-checkpoint', default="", type=str)

args = parser.parse_args()


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
optimizer = optim.SGD(model.parameters(), lr=args.lr_start, momentum=args.momentum)

def train_av(epoch, trainloader, testloader, maxNumBatches):
    print('\nEpoch: %d' % epoch)

    cumulative_loss = 0
    correct = 0
    total_data_samples = 0
    batch_idx = 0




    for (inputs, targets) in enumerate(trainloader):
        if epoch <= 1 and args.lr_increase: #At the start of training, the learning rate is very low and it starts increasing linearly to the desired learning rate
            percentage_lr_decrease = ((epoch * len(trainloader) + batch_idx) / len(trainloader)) / 2
            lr_difference = args.lr - args.lr_start
            current_lr = lr_difference * percentage_lr_decrease + args.lr_start

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr


        if(batch_idx > maxNumBatches):
            break
        torch.set_printoptions(threshold=float('inf'))
        model.train()
        inputs, targets = torch.FloatTensor(targets[0]).to(device), torch.LongTensor(np.array(targets[1])).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        cumulative_loss += loss

        #Before we do the batch update step, we track the train accuracy
        total_data_samples += targets.size(0)

        loss.backward()

        optimizer.step()

        _, predicted = outputs.max(1)

        correct += predicted.eq(targets).sum().item()
        batch_accuracy = predicted.eq(targets).sum().item()

        """#We track the model's accuracy for the test set
        loss_test, correct_test = test_train_confusion(batch_idx, testloader)




        wandb.log({"train_loss": cumulative_loss / (batch_idx+1),
                   "train_accuracy": 100. * correct / total_data_samples,
                   "test_accuracy": 100. * correct_test,
                   "test_loss": loss_test,
                   "epoch:epoch": epoch,
                   "total_data_samples": total_data_samples
                   })"""

        print('Train Batch Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), batch_accuracy, len(inputs),
                batch_accuracy / len(inputs)))
        batch_idx = batch_idx + 1

def test_train_confusion(batch, testloader, msg="Test Train "):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    all_predictions = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    with torch.no_grad():

        for (inputs, targets) in enumerate(testloader):
            inputs, targets = torch.FloatTensor(targets[0]).to(device), torch.LongTensor(np.array(targets[1])).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)


            #This lines of code is used to get the data to use for the confusion matrix
            all_predictions = torch.cat((all_predictions, outputs),dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_idx = batch_idx + 1

        print("Test loss: " + str(test_loss/(batch_idx + 1)) + " ---- Accuracy: " + str(correct/total))

        #calculate the confusion matrix:
        conf_matrix = confusion_matrix(all_targets.cpu(), all_predictions.argmax(dim=1).cpu())
        """
        matrix_size = conf_matrix.shape[0]
        
        # Calculate the starting and ending indices for the center 10x10 matrix
        center_start = (matrix_size // 2) - 10  # Start index for center 10x10
        center_end = center_start + 20  # End index for center 10x10

        # Extract the center 10x10 matrix
        center_matrix = conf_matrix[center_start:center_end, center_start:center_end]

        # Display the center 10x10 matrix
        plt.figure(figsize=(8, 6))
        display = ConfusionMatrixDisplay(confusion_matrix=center_matrix)
        display.plot(cmap=plt.cm.Blues)
        plt.savefig('matrices_plots/confusion_matrix_' + str(batch) + '.png')
        """
        np.save('matrices_numbers/confusion_matrix_' + str(batch) + '.npy', conf_matrix)


    return test_loss/(batch_idx + 1), correct/total







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

        wandb.log({
            "test_loss": test_loss / (batch_idx+1),  # Divide by batch_idx for average loss
            "test_accuracy": 100. * correct / total,  # Divide by total number of samples for accuracy
            "epoch": epoch
        })
    print("Test Epoch "+ str(epoch) +" Loss " + str(test_loss/(batch_idx+1)) + " Accuracy" + str(correct/total))

if(args.continual_type == "CI"):#Class incremental
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

elif(args.continual_type == "CL" ):#Same classes
    split_cifar100 = SplitCIFAR100(
        n_experiences=2,
        seed=1234,
        train_transform = transform_train,
        eval_transform = transform_test,
        first_exp_with_half_classes=False,
        #fixed_class_order=list(range(100)),
        #shuffle=False,
    )
    train_stream = split_cifar100.train_stream
    test_stream = split_cifar100.test_stream

    train_stream_dataset = train_stream[0].dataset
    first_test_dataset = second_test_dataset = test_stream[0].dataset

    total_size = len(train_stream_dataset)
    split_sizes = [int(total_size * 0.5), total_size - int(total_size * 0.5)]  # Splitting into two equal parts

    # Create the two subsets
    first_train_dataset, subset2 = torch.utils.data.random_split(train_stream_dataset, split_sizes)
    second_train_dataset = first_train_dataset + subset2





#split_train_dataset = SplitCIFAR100(n_experiences=2, first_exp_with_half_classes=True, train_transform=transform_train, eval_transform=transform_test, dataset_root='./data')
first_split_trainloader = torch.utils.data.DataLoader(first_train_dataset, batch_size=64, shuffle=True, num_workers=2)
second_split_trainloader = torch.utils.data.DataLoader(second_train_dataset, batch_size=64, shuffle=True, num_workers=2)


# we can recover the corresponding test experience in the test stream
first_split_testloader = torch.utils.data.DataLoader(first_test_dataset, batch_size=64, shuffle=False, num_workers=2)
second_split_testloader = torch.utils.data.DataLoader(second_test_dataset, batch_size=64, shuffle=False, num_workers=2)

#We do a first epoch with the trained path, to see its accuracy

for epoch in range(start_epoch, start_epoch+args.epochs):
    test_train_av(epoch, second_split_testloader)
    train_av(epoch, second_split_trainloader, second_split_testloader, 9999)

torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss,
            }, "./checkpoint/20_classes_trained_model.pth")


