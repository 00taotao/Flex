import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="epochs of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size")
    parser.add_argument('--bs', type=int, default=256, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    # model arguments
    parser.add_argument('--model', type=str, default= 'cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_false', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_false', help='verbose print')

    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--cycles', type=int, default=1, help="cycles of Flex_FL training")
    parser.add_argument('--Forgetting_degree', type=float, default=0.6, help="Forgetting degree")
    parser.add_argument('--num_forget_users', type=int, default=1, help='number of forgetting users')
    parser.add_argument('--epoch_f', type=int, default=100, help="Epoch of Distillation")
    parser.add_argument('--normal_epochs', type=int, default=1, help="epochs of normal local training")
    parser.add_argument('--gamma', type=str, default='1', help="hyperparameter of epoch")
    parser.add_argument('--mu', type=float, default=1, help="hyperparameter1 of loss")
    parser.add_argument('--beta', type=float, default=1, help="hyperparameter2 of loss")

    parser.add_argument('--T', type=float, default=10, help="Temperature of Distillation")
    args = parser.parse_args()
    return args
