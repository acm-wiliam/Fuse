import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN model architecture based on the specified structure:
    - Multiple convolutional layers with ReLU activations
    - Max pooling layers
    - Fully connected layers
    - Designed for CIFAR-10 dataset (32x32x3 input)
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # 第一组卷积层: 输入32x32x3 -> 输出32x32x64
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 第二组卷积层: 输入16x16x64 -> 输出16x16x128
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # 第三组卷积层: 输入8x8x128 -> 输出8x8x256
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # 第四组卷积层: 输入4x4x256 -> 输出4x4x512
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 第一组卷积层
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)  # 32x32x64 -> 16x16x64
        
        # 第二组卷积层
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)  # 16x16x128 -> 8x8x128
        
        # 第三组卷积层
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)  # 8x8x256 -> 4x4x256
        
        # 第四组卷积层
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool(x)  # 4x4x512 -> 2x2x512
        
        # 展平特征图
        x = x.view(-1, 512 * 2 * 2)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 添加Softmax激活函数，dim=1表示在特征维度上进行softmax
        x = F.softmax(x, dim=1)
        
        return x

def get_model(num_classes=10):
    """
    返回初始化好的CNN模型实例
    
    参数:
        num_classes: 分类数量，默认为10（CIFAR-10数据集）
    
    返回:
        初始化好的CNN模型
    """
    return CNN(num_classes=num_classes)

# 示例：如何使用模型
def example_usage():
    # 创建模型
    model = get_model()
    
    # 创建随机输入数据 (batch_size=4, channels=3, height=32, width=32)
    sample_input = torch.randn(4, 3, 32, 32)
    
    # 前向传播
    output = model(sample_input)
    
    print(f"输入形状: {sample_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 模型总结
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")

if __name__ == "__main__":
    example_usage()