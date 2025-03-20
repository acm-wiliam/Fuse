import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse

# 导入自定义模块
from Networks.DNN import get_model
from utils.data_utils import get_cifar10_loaders, get_classes
from utils.train_utils import train_model, evaluate_model
from utils.model_utils import save_model, load_model, get_device
from utils.visualization import (plot_training_history, plot_class_accuracy,
                              plot_confusion_matrix, visualize_model_predictions)


def main(args):
    """
    主函数，包含完整的训练和评估流程
    """
    print("=" * 50)
    print("CIFAR-10 CNN模型训练与评估")
    print("=" * 50)
    
    # 检查设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 设置随机种子，确保结果可复现
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # 数据加载
    print("\n加载CIFAR-10数据集...")
    train_loader, valid_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        augment=args.augment,
        num_workers=args.num_workers
    )
    
    # 创建数据加载器字典（用于训练）
    dataloaders = {
        'train': train_loader,
        'val': valid_loader
    }
    
    print(f"训练集大小: {len(train_loader.sampler)}")
    print(f"验证集大小: {len(valid_loader.sampler)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 获取类别名称
    class_names = get_classes()
    
    # 创建模型
    print("\n创建CNN模型...")
    model = get_model(num_classes=10)
    
    # 如果指定了预训练模型路径，则加载模型
    if args.load_model:
        model = load_model(model, args.load_model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # 训练模型
    if args.train:
        print("\n开始训练模型...")
        model, history = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.epochs,
            device=device
        )
        
        # 可视化训练历史
        if args.visualize:
            print("\n可视化训练历史...")
            plot_training_history(history)
        
        # 保存模型
        if args.save_model:
            print("\n保存模型...")
            # 创建保存目录（如果不存在）
            os.makedirs(args.save_dir, exist_ok=True)
            
            # 保存模型和训练历史
            save_model(model, args.save_dir, metadata={
                'history': history,
                'args': vars(args)
            })
    
    # 在测试集上评估模型
    if args.evaluate:
        print("\n在测试集上评估模型...")
        test_loss, test_acc, class_acc = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device
        )
        
        if args.visualize:
            # 可视化每个类别的准确率
            print("\n可视化每个类别的准确率...")
            plot_class_accuracy(class_acc, class_names)
            
            # 可视化混淆矩阵
            print("\n可视化混淆矩阵...")
            plot_confusion_matrix(model, test_loader, class_names, device)
            
            # 可视化模型预测结果
            print("\n可视化模型预测结果...")
            visualize_model_predictions(model, test_loader, class_names, device)
            
    print("\n程序执行完成!")


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN模型训练与评估')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=128, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--lr-step', type=int, default=30, help='学习率调整步长')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='学习率衰减因子')
    
    # 数据参数
    parser.add_argument('--valid-size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--augment', action='store_true', default=True, help='是否使用数据增强')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载线程数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save-model', action='store_true', default=True, help='是否保存模型')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--load-model', type=str, default='', help='预训练模型路径')
    parser.add_argument('--train', action='store_true', default=True, help='是否训练模型')
    parser.add_argument('--evaluate', action='store_true', default=True, help='是否评估模型')
    parser.add_argument('--visualize', action='store_true', default=True, help='是否可视化结果')
    
    args = parser.parse_args()
    
    main(args)