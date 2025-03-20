import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, 
                num_epochs=25, device='cuda', verbose=True):
    """
    训练PyTorch模型
    
    参数:
        model: 待训练的模型
        dataloaders: 包含'train'和'val'键的字典，对应训练和验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        num_epochs: 训练轮数
        device: 训练设备 ('cuda' 或 'cpu')
        verbose: 是否打印训练进度
        
    返回:
        model: 训练好的模型
        history: 包含训练和验证损失及准确率的字典
    """
    since = time.time()
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    # 保存最佳模型权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            loader = dataloaders[phase]
            if verbose:
                loader = tqdm(loader, desc=f'{phase}', leave=False)
                
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度归零
                optimizer.zero_grad()
                
                # 前向传播
                # 只在训练时记录梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是训练阶段，则反向传播和参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            # 计算epoch损失和准确率
            epoch_loss = running_loss / len(dataloaders[phase].sampler)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].sampler)
            
            # 记录训练历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if verbose:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 深拷贝模型（如果是目前为止最好的）
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        if verbose:
            print()
    
    time_elapsed = time.time() - since
    if verbose:
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, dataloader, criterion, device='cuda', verbose=True):
    """
    评估模型在测试集上的性能
    
    参数:
        model: 要评估的模型
        dataloader: 测试数据加载器
        criterion: 损失函数
        device: 评估设备 ('cuda' 或 'cpu')
        verbose: 是否打印评估结果
        
    返回:
        test_loss: 测试损失
        test_acc: 测试准确率
        class_accuracy: 每个类别的准确率字典
    """
    model = model.to(device)
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    # 用于计算每个类别的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    # 不计算梯度
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing', disable=not verbose):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 计算每个类别的准确率
            correct = (preds == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    # 计算总体损失和准确率
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)
    
    # 计算每个类的准确率
    class_accuracy = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 
                      for i in range(10)}
    
    if verbose:
        print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
        
        # 打印每个类别的准确率
        for i in range(10):
            print(f'Accuracy of class {i}: {100 * class_accuracy[i]:.2f}%')
    
    return test_loss, test_acc.item(), class_accuracy