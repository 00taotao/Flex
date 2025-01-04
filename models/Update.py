import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset# 输入数据集
        self.idxs = list(idxs)# 输入索引号

    def __len__(self):
        return len(self.idxs)# 图片数量

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]#按照索引号号分配图片和标签
        return image, label


class Local_Update(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args # 输入参数
        self.loss_func = nn.CrossEntropyLoss()#损失函数用交叉熵函数
        self.selected_clients = [] #客户端
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True) #取出图片

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.normal_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 100 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter+1, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def split(self,net,priamrynet,subnet):
        priamrynet.load_state_dict(net, strict=False)
        subnet.load_state_dict(net, strict=False)
        #返回网络参数
        return priamrynet.state_dict(), subnet.state_dict()

class LocalUpdate_A(object):
    def __init__(self, args, dataset=None, idxs=None, Forgetting_degree=0.5):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.Forget_degree = Forgetting_degree
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.contribution_degree = 1 - Forgetting_degree
        self.batch_num = len(self.ldr_train.dataset) / self.args.local_bs
        self.batch_num_forget = int(args.gamma * self.batch_num * self.Forget_degree)
        self.batch_num_remain = int(args.gamma * self.batch_num * self.contribution_degree)
        self.loss_beta = args.beta
        self.loss_mu = args.mu
    # 遗忘网络训练
    def Forget_train(self, Remainnet, Forgetnet, optimizer_Forget, prev_loss):
        Forgetnet.train()
        Remainnet.train()
        # 如果遗忘轮数为0，返回无限大
        if self.batch_num_forget == 0:
            return float('inf'), float('inf'), float('inf')
        # 记录初始网络参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        last_epoch_loss = []
        last_epoch_loss_avg = 0
        batch_sum = 0
        while batch_sum < self.batch_num_forget:
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_sum > self.batch_num_forget:
                    break
                batch_sum += 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Forget.step()
                if self.args.verbose and batch_idx % 5 == 0:
                    print('Forget Update batch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_sum,self.batch_num_forget , batch_idx * len(images), len(self.ldr_train.dataset),
                                             100. * batch_idx / len(self.ldr_train), loss.item()))
                last_epoch_loss.append(loss.item())
        # 计算最后一轮平均损失值
        last_epoch_loss_avg = sum(last_epoch_loss) / len(last_epoch_loss)

        # 记录训练后网络参数
        Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])

        # 计算模型差异度
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_sim = cos(w_glob, w_train)
        cos_sim = (cos_sim + 1.0) / 2.0
        diff = 1 - cos_sim

        # 计算差异阈值
        diff_threshold = self.contribution_degree / self.Forget_degree * diff
        # 计算损失阈值
        loss_threshold = self.contribution_degree / self.Forget_degree * self.loss_beta*(last_epoch_loss[0] - self.loss_mu * last_epoch_loss[-1])
        # 返回模型差异度阈值和损失值差异阈值
        return diff_threshold, loss_threshold, last_epoch_loss_avg

    def record(self, Remainnet, Forgetnet):
        record_loss = []
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            temp = Remainnet(images)
            log_probs = Forgetnet(temp)
            record_loss.append(self.loss_func(log_probs, labels).item())
        return sum(record_loss) / len(record_loss)

    # 保留网络训练
    def Remain_train(self, Remainnet, Forgetnet, optimizer_Remain, diff_threshold, loss_threshold,
                     prev_loss):
        # 记录初始网路参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        # 训练
        Remainnet.train()
        Forgetnet.train()
        Remainnet_avg_loss = 0
        batch_sum = 0
        while batch_sum < self.batch_num_remain:
            batch_loss = []
            # 存储网络参数
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                tmp_model = copy.deepcopy(Remainnet.state_dict())
                tmp_loss = copy.deepcopy(Remainnet_avg_loss)
                if batch_sum >= self.batch_num_remain:
                    return Remainnet_avg_loss
                batch_sum += 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Remain.step()
                batch_loss.append(loss.item())

                # 记录平均损失值
                Remainnet_avg_loss = sum(batch_loss) / len(batch_loss)
                # loss_diff = prev_loss - Remainnet_avg_loss
                loss_diff = self.loss_beta * (batch_loss[0] - self.loss_mu * batch_loss[-1])
                # 从第5次判断是否达到损失值阈值
                if batch_idx > 1 and loss_diff > loss_threshold:
                    print("loss_difff:{:.4f} > loss_thresholdf:{:.4f}".format(loss_diff, loss_threshold))
                    Remainnet.load_state_dict(tmp_model)
                    Remainnet_avg_loss = tmp_loss
                    return Remainnet_avg_loss

                # 记录训练完成后网络参数
                Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
                Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
                w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])
                # 差异度
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sim = cos(w_glob, w_train)
                cos_sim = (cos_sim + 1.0) / 2.0
                diff = 1 - cos_sim
                # 判断是否达到差异度阈值，跳出当前循环
                if diff > diff_threshold:
                    print("diff:{:.4f}> diff_thresholdf:{:.4f}".format(diff, diff_threshold))
                    Remainnet.load_state_dict(tmp_model)
                    Remainnet_avg_loss = tmp_loss
                    return Remainnet_avg_loss
                if self.args.verbose and batch_idx % 5 == 0:
                    print(
                        'Remainnet Update batch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f} diff/diff_threshold:{:.4f}/{:.4f}\tloss_diff/loss_threshold:{:.4f}/{:.4f}'.format(
                            batch_sum,self.batch_num_remain, batch_idx * len(images), len(self.ldr_train.dataset),
                                                 100. * batch_idx / len(self.ldr_train), loss.item(), diff,
                            diff_threshold, loss_diff, loss_threshold))
        return Remainnet_avg_loss

    def Localtrain(self, Remainnet, Forgetnet):
        # 初始化优化器
        optimizer_Remain = torch.optim.SGD(Remainnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_Forget = torch.optim.SGD(Forgetnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 先记录初始损失值
        prev_loss = self.record(Remainnet, Forgetnet)
        print('prev_loss:{:.4f}'.format(prev_loss))
        loss = []
        for cycle in range(self.args.cycles):
            print("Cycle: {}".format(cycle + 1))
            # 训练遗忘网络
            print("Training Forgetnet")
            diff_threshold, loss_threshold, loss_train = self.Forget_train(Remainnet,Forgetnet, optimizer_Remain,prev_loss)
            if loss_train != float('inf'):
                loss.append(loss_train)
            # 差异度阈值， 损失值阈值 ，当前平均损失值
            print("diff_threshold: {:.4f}, loss_threshold: {:.4f}, Loss: {:.4f}".format(diff_threshold, loss_threshold, loss_train))
            # 再训练子网络
            print("Training Remainnet")
            prev_loss = self.Remain_train(Remainnet, Forgetnet, optimizer_Forget,diff_threshold, loss_threshold, loss_train)
            loss.append(prev_loss)
        # 计算平均损失值
        loss_avg = sum(loss) / len(loss)
        # 返回网络和损失值
        return Remainnet.state_dict(), Forgetnet.state_dict(), loss_avg



