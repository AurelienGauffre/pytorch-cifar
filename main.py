'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='batch size for training')
parser.add_argument('--num_workers', '-nw', default=2, type=int, help='num worker for train and test dataloader')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--cifar_binary', action='store_true',
                    help='If true, uses the cifar dataset from torchvision')
parser.add_argument('--wandb', action='store_false',
                    help='wandb logging')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data

random_crop = transforms.RandomSizedCrop(32)
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

# train_transform = transforms.Compose([
#     random_crop,
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize,
# ])
#
# test_transform = transforms.Compose([
#     random_crop,
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize,
# ])



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

# Cifar
if not args.cifar_binary:
    print('Using Cifar .png images ...')

    trainset = torchvision.datasets.ImageFolder(root=os.path.join('~/datasets', 'cifar10im', 'train'),
                                            transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root=os.path.join('~/datasets', 'cifar10im', 'val'),
                                               transform=test_transform)
else :
    print('Using Cifar BINARIES from torchvision ...')
    trainset = torchvision.datasets.CIFAR10(
       root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(
       root='./data', train=False, download=True, transform=test_transform)

print('==> Preparing data..')


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')

net = ResNet18()
#net = torchvision.models.resnet18(pretrained=False, num_classes=10)

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()


        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (len(trainloader)), correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        return test_loss / (len(testloader) + 1), correct / total
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

if args.wandb :
    wandb.init(project='NAS-SSL-MTL', entity='aureliengauffre',
               group='Debug')
    wandb.run.name = 'Cifar10 VS Cifar10im'

print('NB PARAMS', sum(p.numel() for p in net.parameters()))
for epoch in range(start_epoch + 1, start_epoch + 201):
    print('starting')

    train_loss, train_acc = train(epoch)

    test_loss, test_acc = test(epoch)
    scheduler.step()
    main_log_dic = {'epoch': epoch + 1,
                    'vanilla train loss': train_loss,
                    'vanilla train accuracy': train_acc,
                    'vanilla val loss': test_loss,
                    'vanilla val accuracy': test_acc,
                    }
    if args.wandb:
        wandb.log(main_log_dic)
    #wandb.watch(net, criterion, log="all")
