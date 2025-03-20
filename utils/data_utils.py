import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import DataLoader, SubsetRandomSampler

def get_cifar10_loaders(batch_size=128, valid_size=0.1, augment=True, 
                         num_workers=2, pin_memory=True):
    """
    加载CIFAR-10数据集并返回训练、验证和测试数据的DataLoader
    
    参数:
        batch_size: 每个batch的样本数
        valid_size: 验证集比例，从训练集中分离
        augment: 是否进行数据增强
        num_workers: 数据加载的工作线程数
        pin_memory: 是否将数据加载到CUDA固定内存中，加速GPU训练
        
    返回:
        train_loader: 训练集DataLoader
        valid_loader: 验证集DataLoader
        test_loader: 测试集DataLoader
    """
    # 定义数据预处理步骤
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    # 训练数据转换 (包括数据增强)
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # 测试数据转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # 获取数据集根目录的绝对路径
    # 使用项目根目录下的data文件夹
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # 加载数据集，设置download=False使用本地数据
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, 
        # root = './data',
        train=True,
        download=True, 
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, 
        # root = './data',
        train=False,
        download=True, 
        transform=test_transform
    )
    
    # 从训练集中创建验证集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, valid_loader, test_loader


def get_classes():
    """返回CIFAR-10的类名称列表"""
    return ['飞机', '汽车', '鸟', '猫', '鹿', 
            '狗', '蛙', '马', '船', '卡车']