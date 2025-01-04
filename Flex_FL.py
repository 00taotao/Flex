# Description: This file is used to implement the Flex-FL framework.
import copy
from torchvision import datasets, transforms
import torch
from utils.sample import mnist_noniid, fmnist_noniid, cifar_noniid, cifar100_noniid, mnist_iid, cifar_iid, fmnist_iid
from utils.options import args_parser
from models.Update import LocalUpdate_A, Local_Update
from models.nets import Mnistprimarynet, Mnistsubnet, ResNet18primary, ResNet18subnet, ResBlock, VGG11, VGG11primary, \
    VGG11subnet, LeNet5primary, LeNet5subnet, CNNMnist, LeNet5, ResNet18
from models.FedAvg import FedAvg_noniid
from models.Test import test_img
import os

if __name__ == '__main__':
    # parse args
    print("-------Initialize parameters-------")
    args = args_parser()
    args.dataset = 'fmnist'
    args.model = 'LeNet5'
    args.epochs = 100
    args.num_users = 10
    args.lr = 0.001
    args.iid = False
    args.normal_epochs = 5
    args.cycles = 5
    args.gamma = 1.2
    args.beta = 0.95
    args.mu = 0.95
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    print("-------Allocate dataset-------")
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = fmnist_noniid(dataset_train, args.num_users)
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'cifar':
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

        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2860,) ,(0.3205,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True, transform=transform)
        # sample users 分配数据集
        if args.iid:
            dict_users = fmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = fmnist_noniid(dataset_train, args.num_users)
        net_glob = LeNet5().to(args.device)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users)
        net_glob = ResNet18(ResBlock).to(args.device)
    else:
        exit('Error: unrecognized dataset')


    # build model
    print("-------Initialize network-------")
    if args.model == 'cnn' and args.dataset == 'mnist':
        Remainnet = Mnistprimarynet(args=args).to(args.device)
        Forgetnet = Mnistsubnet(args=args).to(args.device)
        w_glob_Remain = Remainnet.state_dict()
        w_glob_Forget = Forgetnet.state_dict()
    elif args.model == 'ResNet18' and args.dataset == 'cifar':
        Remainnet = ResNet18primary(ResBlock).to(args.device)
        Forgetnet = ResNet18subnet(ResBlock).to(args.device)
        w_glob_Remain = Remainnet.state_dict()
        w_glob_Forget = Forgetnet.state_dict()
    elif args.model == 'VGG11' and args.dataset == 'cifar':
        Remainnet = VGG11primary().to(args.device)
        Forgetnet = VGG11subnet().to(args.device)
        w_glob_Remain = Remainnet.state_dict()
        w_glob_Forget = Forgetnet.state_dict()
    elif args.model == 'LeNet5' and args.dataset == 'fmnist':
        Remainnet = LeNet5primary().to(args.device)
        Forgetnet = LeNet5subnet().to(args.device)
        w_glob_Remain = Remainnet.state_dict()
        w_glob_Forget = Forgetnet.state_dict()

    Forgetnet_history = copy.deepcopy(w_glob_Forget)
    # training
    print("-------Training-------")
    # unlearning client
    idxs_users = range(args.num_users)
    client_data_sizes = [len(dict_users[i]) for i in range(args.num_users)]
    m = args.num_forget_users
    idxs_users_Forget = [idxs_users[x]for x in range(m)]
    if args.all_clients:
        w_locals_Remain = [w_glob_Remain for i in range(args.num_users)]
        w_locals_Forget = [w_glob_Forget for i in range(args.num_users)]
    # test accuracy
    acc_test, loss_test = test_img(Remainnet,Forgetnet, dataset_test, args)


    # 本地训练
    for iter in range(args.epochs):
        print('-------Rounds {}-------'.format(iter))
        loss_locals = []
        if not args.all_clients:
            w_locals_Remain = []
            w_locals_Forget = []
        for idx in idxs_users:
            print("-------Client {} is training-------".format(idx))
            if idx in idxs_users_Forget:
                print("-------Client {} is unlearning client-------".format(idx))
                local = LocalUpdate_A(args=args, dataset=dataset_train, idxs=dict_users[idx], Forgetting_degree = args.Forgetting_degree)
                w_Remain, w_Forget,loss = local.Localtrain(copy.deepcopy(Remainnet),copy.deepcopy(Forgetnet))
            else:
                local = Local_Update(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_Remain, w_Forget= local.split(w, copy.deepcopy(Remainnet),copy.deepcopy(Forgetnet))
            if args.all_clients:
                w_locals_Remain[idx] = copy.deepcopy(w_Remain)
                w_locals_Forget[idx] = copy.deepcopy(w_Forget)
            else:
                w_locals_Remain.append(copy.deepcopy(w_Remain))
                w_locals_Forget.append(copy.deepcopy(w_Forget))
            loss_locals.append(copy.deepcopy(loss))
            print("Client {} training loss: {:.4f}".format(idx, loss))
        # update global weights
        print("-------Update network-------")
        w_glob_Remain = FedAvg_noniid(w_locals_Remain,client_data_sizes)
        w_glob_Forget = FedAvg_noniid(w_locals_Forget,client_data_sizes)
        # copy weight to net_glob
        Remainnet.load_state_dict(w_glob_Remain)
        Forgetnet.load_state_dict(w_glob_Forget)
        net_glob.load_state_dict(w_glob_Remain, strict=False)
        net_glob.load_state_dict(w_glob_Forget, strict=False)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        acc_test, loss_test = test_img(Remainnet,Forgetnet, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
    # Save a model
    print("-------Save a model-------")
    if not os.path.exists('./save/{}'.format(args.dataset)):
        os.makedirs('./save/{}'.format(args.dataset))
    torch.save(Remainnet.state_dict(), './save/{}/Remainnet_{}_{}_{}_C{}_N{}_{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs, args.num_users, args.num_forget_users,args.Forgetting_degree))
    torch.save(Forgetnet.state_dict(), './save/{}/Forgetnet_{}_{}_{}_C{}_N{}_{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs,args.num_users, args.num_forget_users,args.Forgetting_degree))
    torch.save(Forgetnet_history, './save/{}/Forgetnet_history_{}_{}_{}_C{}_N{}_{}.pth'.format(args.dataset,args.model,args.dataset,args.epochs,args.num_users, args.num_forget_users,args.Forgetting_degree))
