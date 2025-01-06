# Flex(Federated Unlearning)
## About The Project
Flex is used to meet clients' different forgetting preferences and enable arbitrary forgetting.
## Presented Unlearning Methods
- **Multi-head training** : The Flex method divides federated learning contributions into computing
power contribution, accuracy contribution, and robustness contribution based on
the contribution definition, and uses multi-head training to allocate the
contributions into two parts of the model. Part is used to forget the contribution, and part is used to keep 
the contribution.
- **Multi-head Recovery** : The Flex method first resets the forgotten part, and then uses the multi head distillation method 
to distill the two parts of the network in different proportions to achieve rapid recovery.


## Getting Started
### Requirements
| Package     | Version      |
|-------------|--------------|
| torch       | 1.12.1+cu113 |
| torchvision | 1.13.1+cu113 |
| Pillow      | 11.0.0       |
| python      | 3.10.12      |
| numpy       | 1.23.5       |

### File Structure
```
├─data
│    └─ datasets.txt
│      
├─models
│    ├─  Distillation.py
│    ├─  FedAvg.py
│    ├─  nets.py
│    ├─  Test.py
│    └─  Update.py
│ 
│          
├─utils
│    ├─  FD_datasets.py
│    ├─  options.py
│    └─  sample.py
│    
├─ Flex_FL.py
├─ Flex_FUL.py
└─ README.md
```
There are severl parts in the code:
- data folder: This folder contains the training and testing data forthe target model. In order to reduce the memory space, we just list thelinks to theset dataset here.

    -- EMNIST download link: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
    
    -- Cloth dataset download link: https://github.com/alexeygrigorev/clothing-dataset/archive/refs/heads/master.zip
    
    -- Imagenet-tiny download link: https://cs231n.stanford.edu/tiny-imagenet-200.zip
    
    The rest is the dataset that comes with PyTorch(MNIST, FMNIST, CIFAR-10, CIFAR-100).
             
- models folder: This folder contains the implementation of the target model, the federated learning algorithm, and the unlearning algorithm. 

  -- The target model is implemented in the nets.py file. 
  
  -- The federated learning aggregation algorithm is implemented in the FedAvg.py file. 
  
  -- The federated learning algorithm is implemented in the Update.py file. 

  -- The distillation algorithm is implemented in the Distillation.py file. 

  -- The Test.py file is used to test the performance of the target model.

- utils folder: This folder contains the implementation of the dataset loader, the options parser, and the sample loader. 

  -- The FD_datasets.py file is used to load the dataset. 

  -- The options.py file is used to parse the options. 

  -- The sample.py file is used to load the sample.
- Flex_FL.py: This file is used to train the target model using the federated learning algorithm.
- Flex_FUL.py: This file is used to train the target model using the federated unlearning algorithm.

## Parameter Setting of Flex
- **--epochs**: The number of FL training epochs. The default value is 100.
- **--num_users**: The number of users in the federated learning system. The default value is 10.
- **--frac**: The fraction of clients used in each training round. The default value is 1.
- **--local_ep**: The number of local epochs per client. The default value is 5.
- **--local_bs**: The local batch size for training. The default value is 128.
- **--bs**: The batch size for testing. The default value is 256.
- **--lr**: The learning rate. The default value is 0.001.
- **--momentum**: The momentum for SGD optimizer. The default value is 0.9.
- **--split**: The type of train-test split. The default value is 'user'.
- **--model**: The model name. The default value is 'cnn'.
- **--kernel_num**: The number of each kind of kernel. The default value is 9.
- **--kernel_sizes**: Comma-separated kernel sizes for convolution. The default value is '3,4,5'.
- **--norm**: The normalization method. Options are 'batch_norm', 'layer_norm', or None. The default value is 'batch_norm'.
- **--num_filters**: The number of filters for convolutional networks. The default value is 32.
- **--max_pool**: Whether to use max pooling instead of strided convolutions. The default value is 'True'.
- **--dataset**: The name of the dataset. The default value is 'mnist'.
- **--iid**: Specifies whether the data is independent and identically distributed (i.i.d). The default value is False.
- **--num_classes**: The number of classes in the dataset. The default value is 10.
- **--num_channels**: The number of channels in the images. The default value is 1.
- **--gpu**: The GPU ID to use, -1 for CPU. The default value is 0.
- **--stopping_rounds**: The number of rounds for early stopping. The default value is 10.
- **--verbose**: Whether to use verbose printing. The default value is False.
- **--all_clients**: Whether to aggregate over all clients. The default value is True.
- **--cycles**: The number of cycles of Flex_FL training. The default value is 1.
- **--Forgetting_degree**: The degree of forgetting. The default value is 0.6.
- **--num_forget_users**: The number of users to forget. The default value is 1.
- **--epoch_f**: The number of epochs for distillation. The default value is 100.
- **--normal_epochs**: The number of epochs for normal local training. The default value is 1.
- **--gamma**: A hyperparameter for epochs. The default value is 1.
- **--mu**: The first hyperparameter for loss. The default value is 1.
- **--beta**: The second hyperparameter for loss. The default value is 1.
- **--T**: The temperature for distillation. The default value is 10.

## Execute Flex
Edit Flex_FL.py or Flex_FUL.py files, modify parameters, such as datasets, epochs, and model, and run.


