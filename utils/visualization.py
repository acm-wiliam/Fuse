import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools


def plot_training_history(history, figsize=(12, 5)):
    """
    绘制训练历史，包括损失和准确率
    
    参数:
        history: 包含训练历史数据的字典
        figsize: 图形尺寸
    """
    plt.figure(figsize=figsize)
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_class_accuracy(class_accuracy, class_names=None, figsize=(10, 6)):
    """
    绘制每个类别的准确率条形图
    
    参数:
        class_accuracy: 每个类别准确率的字典
        class_names: 类名列表（可选）
        figsize: 图形尺寸
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(class_accuracy))]
    
    plt.figure(figsize=figsize)
    
    # 准备绘图数据
    classes = list(class_accuracy.keys())
    accuracy = [class_accuracy[cls] * 100 for cls in classes]
    
    # 绘制条形图
    bars = plt.bar(range(len(classes)), accuracy, align='center')
    plt.xticks(range(len(classes)), [class_names[i] for i in classes], rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.title('每个类别的模型准确率')
    plt.ylim(0, 105)  # 设置y轴范围，留出空间显示数值标签
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataloader, class_names=None, device='cuda', figsize=(10, 8)):
    """
    生成并绘制混淆矩阵
    
    参数:
        model: 训练好的模型
        dataloader: 测试数据加载器
        class_names: 类名列表（可选）
        device: 计算设备
        figsize: 图形尺寸
    """
    # 预测
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=figsize)
    
    # 归一化混淆矩阵（按行）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 使用热图绘制
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('归一化混淆矩阵')
    plt.tight_layout()
    plt.show()
    
    return cm


def visualize_model_predictions(model, dataloader, class_names, device='cuda', num_images=16):
    """
    可视化模型预测结果
    
    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        class_names: 类名列表
        device: 计算设备
        num_images: 要显示的图像数量
    """
    model = model.to(device)
    model.eval()
    
    # 获取一批数据
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # 预测
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # 绘制图像和预测结果
    plt.figure(figsize=(15, 12))
    
    for idx in range(min(num_images, len(images))):
        ax = plt.subplot(4, num_images//4, idx + 1)
        img = images[idx].numpy().transpose((1, 2, 0))
        
        # 反归一化
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # 标记预测结果（绿色表示正确，红色表示错误）
        title_color = 'green' if preds[idx] == labels[idx] else 'red'
        ax.set_title(f'预测: {class_names[preds[idx]]}\n真实: {class_names[labels[idx]]}',
                   color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()