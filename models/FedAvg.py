import copy
import torch

# 联邦聚合
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_noniid(w, client_data_sizes):
    """
    Args:
    w (list): List of client weights.
    client_data_sizes (list): List of data sizes for each client.

    Returns:
    w_avg: Aggregated model weights.
    """
    total_data = sum(client_data_sizes)  # 计算总的数据量
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])

    for i in range(len(w)):
        for k in w_avg.keys():
            w_avg[k] += w[i][k] * (client_data_sizes[i] / total_data)

    return w_avg

def FedAvg_noniid_float(w, client_data_sizes):
    """
    Args:
    w (list): List of client weights.
    client_data_sizes (list): List of data sizes for each client.

    Returns:
    w_avg: Aggregated model weights.
    """
    total_data = float(sum(client_data_sizes))  # 计算总的数据量
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k], dtype=torch.float32)

    for i in range(len(w)):
        for k in w_avg.keys():
            w_avg[k] += w[i][k].float() * (float(client_data_sizes[i])/ total_data)

    return w_avg

def minus_value(w1,w2):
    w_avg = copy.deepcopy(w1)
    for k in w_avg.keys():
        w_avg[k] = w1[k] - w2[k]
    return w_avg

def add_value(w1,w2):
    w_avg = copy.deepcopy(w1)
    for k in w_avg.keys():
        w_avg[k] = w1[k] + w2[k]
    return w_avg
