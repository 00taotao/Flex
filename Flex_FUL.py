# Description: This file is used to implement the Flex-FUL framework.
from utils.FD_datasets import EMNISTDataset, ClothingDataset_whole, TinyImageNet
import copy
from torchvision import datasets, transforms
import torch
from utils.options import args_parser
from models.nets import *
from models.Test import test_img
from models.Distillation import Distill
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    # parse args
    print("-------Initialize parameters-------")
    args = args_parser()
    args.dataset = 'fmnist'
    args.model = 'LeNet5'
    args.iid = False
    args.normal_epochs = 5
    args.epochs = 100
    args.epoch_f = 100
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # load dataset
    print("---------Load dataset---------")
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        dataset_FD = EMNISTDataset('./data/emnist/EMNIST/raw/', split='balanced', train=True, transform=trans_mnist)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform_test)
        dataset_FD = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transform_train)
    elif args.dataset == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,) ,(0.3205,))])
        transform2 = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.2860,) ,(0.3205,))])
        dataset_test = datasets.FashionMNIST('./data/fmnist/', train=False, download=True, transform=transform)
        dataset_FD = ClothingDataset_whole('./data/clothing-dataset-master/images', transform=transform2)
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            # 图片从64*64缩小到32*32
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),  # R,G,B每层的归一化用到的均值和方差
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

        # dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=transform_train)
        dataset_FD = TinyImageNet('./data/tiny-imagenet-200', train=True, transform=transform)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test)
    else:
        exit('Error: unrecognized dataset')



    # Load network parameters
    Remainnet = torch.load('./save/{}/Remainnet_{}_{}_{}_C{}_N{}_{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs, args.num_users, args.num_forget_users,args.Forgetting_degree))
    Forgetnet_final = torch.load('./save/{}/Forgetnet_{}_{}_{}_C{}_N{}_{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs,args.num_users, args.num_forget_users,args.Forgetting_degree))
    Forgetnet_history = torch.load('./save/{}/Forgetnet_history_{}_{}_{}_C{}_N{}_{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs,args.num_users, args.num_forget_users,args.Forgetting_degree))



    print("-------Recovery phase-------")
    #Create a new model and load the Remainnet and Forgetnet_history parameters
    if args.model == 'ResNet18' and args.dataset == 'cifar100':
        Remainnet_F = ResNet18primary(ResBlock).to(args.device)
        Forgetnet_F = ResNet18subnet(ResBlock).to(args.device)
        Forgetnet = copy.deepcopy(Forgetnet_F)
        Remainnet_F.load_state_dict(Remainnet)
        T_Remainnet_F = copy.deepcopy(Remainnet_F)
        Forgetnet_F.load_state_dict(Forgetnet_history)
        Forgetnet.load_state_dict(Forgetnet_final)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        Remainnet_F = Mnistprimarynet(args=args).to(args.device)
        Forgetnet_F = Mnistsubnet(args=args).to(args.device)
        Forgetnet = copy.deepcopy(Forgetnet_F)
        Remainnet_F.load_state_dict(Remainnet)
        T_Remainnet_F = copy.deepcopy(Remainnet_F)
        Forgetnet_F.load_state_dict(Forgetnet_history)
        Forgetnet_F.load_state_dict(Forgetnet_history)
        Forgetnet.load_state_dict(Forgetnet_final)
    elif args.model == 'VGG11' and args.dataset == 'cifar10':
        Remainnet_F = VGG11primary().to(args.device)
        Forgetnet_F = VGG11subnet().to(args.device)
        Forgetnet = copy.deepcopy(Forgetnet_F)
        Remainnet_F.load_state_dict(Remainnet)
        T_Remainnet_F = copy.deepcopy(Remainnet_F)
        Forgetnet_F.load_state_dict(Forgetnet_history)
        Forgetnet.load_state_dict(Forgetnet_final)
    elif args.model == 'LeNet5' and args.dataset == 'fmnist':
        Remainnet_F = LeNet5primary().to(args.device)
        Forgetnet_F = LeNet5subnet().to(args.device)
        Forgetnet = copy.deepcopy(Forgetnet_F)
        Remainnet_F.load_state_dict(Remainnet)
        T_Remainnet_F = copy.deepcopy(Remainnet_F)
        Forgetnet_F.load_state_dict(Forgetnet_history)
        Forgetnet.load_state_dict(Forgetnet_final)

    # Test the performance of the teacher model
    acc_test, loss_test = test_img(Remainnet_F,Forgetnet, dataset_test, args)
    print("Teacher testing accuracy: {:.2f}".format(acc_test))
    # Test the performance of student models
    acc_test, loss_test = test_img(Remainnet_F,Forgetnet_F, dataset_test, args)
    print("Student testing accuracy: {:.2f}".format(acc_test))


    # Initialize distillation function
    args.T = 10
    distill = Distill(args=args)
    # Optimizer settings
    optimizer_Forget = torch.optim.SGD(Forgetnet_F.parameters(), lr=0.001, momentum=args.momentum)
    optimizer_Remain = torch.optim.SGD(Remainnet_F.parameters(), lr=0.001, momentum=args.momentum)
    dataset_boost = DataLoader(dataset_FD, batch_size=128, shuffle=True)
    # Set the model to evaluation mode
    Forgetnet.eval()
    T_Remainnet_F.eval()
    #Training
    for epoch in range(args.epoch_f):
        Remainnet_F.train()
        Forgetnet_F.train()
        # Reduce distillation temperature
        if epoch % 10 == 9 and args.T > 2:
            args.T -= 1
            distill = Distill(args=args)
        for iter in range(args.normal_epochs):
            for batch_idx, images in enumerate(dataset_boost):
                images = images.to(args.device)
                optimizer_Forget.zero_grad()
                optimizer_Remain.zero_grad()
                temp = Remainnet_F(images)
                temp_ = T_Remainnet_F(images)
                outputs = Forgetnet_F(temp)
                teacher_outputs = Forgetnet(temp_)
                loss = distill.distillation_loss_zero(outputs, teacher_outputs = teacher_outputs)
                loss.backward()
                optimizer_Forget.step()
                optimizer_Remain.step()
                if args.verbose and batch_idx % 100 == 0:
                    print('Distillation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(dataset_boost.dataset),
                            100. * batch_idx / len(dataset_boost), loss.item()))
        # Test model performance
        acc_test, loss_test = test_img(Remainnet_F,Forgetnet_F, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        if not os.path.exists('./save/{}'.format(args.dataset)):
            os.makedirs('./save/{}'.format(args.dataset))
        torch.save(Remainnet_F.state_dict(), './save/{}/Remainnet_F_{}_{}_{}_C{}_iid{}_F{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs, args.num_users, args.iid,args.Forgetting_degree))
        torch.save(Forgetnet_F.state_dict(), './save/{}/Forgetnet_F_{}_{}_{}_C{}_iid{}_F{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs, args.num_users, args.iid,args.Forgetting_degree))

