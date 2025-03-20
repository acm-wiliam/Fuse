import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可重复
torch.manual_seed(0)
np.random.seed(0)

# MNIST 数据集的类别名称
MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def mnist_model(pretrained=False, device="cpu"):
    """构建MNIST模型

    参数:
    pretrained (bool): 如果为True，返回预训练模型
    device (str): 模型所在设备，'cpu'或'cuda'
    
    返回:
    Net: MNIST模型实例
    """
    model = Net()
    
    # 如果使用预训练模型
    if pretrained:
        # 尝试从不同可能的路径加载预训练权重
        try:
            # 先尝试从Fuse/data/state_dicts目录加载
            script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'state_dicts'))
            state_dict_path = os.path.join(script_dir, "mnist_cnn.pt")
            
            # 加载预训练权重
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"成功加载预训练权重: {state_dict_path}")

        except Exception as e:
            print(f"无法加载预训练权重: {e}")
            print("将使用随机初始化的模型")
    
    return model

# 数据预处理的均值和标准差
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

def prepare_data(batch_size=64, num_workers=2):
    """准备MNIST数据集的数据加载器
    
    参数:
    batch_size (int): 批量大小
    num_workers (int): 数据加载的工作线程数
    
    返回:
    DataLoader: 测试数据加载器
    """
    # 测试数据变换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    
    # 尝试不同的数据路径
    data_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),  # Fuse/data
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')  # 上级目录的data
    ]
    
    # 尝试每个路径直到成功
    test_dataset = None
    for data_path in data_paths:
        try:
            print(f"尝试从路径加载数据: {data_path}")
            test_dataset = datasets.MNIST(
                root=data_path,
                train=False,
                download=False,  # 先尝试不下载
                transform=transform_test
            )
            print(f"成功从{data_path}加载测试集")
            break
        except (RuntimeError, FileNotFoundError) as e:
            print(f"无法从{data_path}加载数据: {e}")
    
    # 如果所有路径都失败，尝试下载数据集
    if test_dataset is None:
        try:
            print("尝试下载MNIST数据集...")
            test_dataset = datasets.MNIST(
                root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
                train=False,
                download=True,
                transform=transform_test
            )
            print("成功下载MNIST测试集")
        except Exception as e:
            raise RuntimeError(f"无法加载或下载MNIST数据集: {e}")
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 不打乱数据顺序
        num_workers=num_workers,  # 数据加载的工作线程数
        pin_memory=True,  # 将数据加载到CUDA固定内存，加速数据传输到GPU
    )
    
    return test_loader

def test(model, device, test_loader):
    """评估模型在测试集上的性能
    
    参数:
    model (nn.Module): 待评估的模型
    test_loader (DataLoader): 测试数据加载器
    device (torch.device): 用于评估的设备
    
    返回:
    float: 测试损失值
    float: 测试准确率
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 合计批量损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    return test_loss, accuracy

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = mnist_model(pretrained=True, device=device)
    model = model.to(device)
    
    try:
        # 准备数据
        test_loader = prepare_data(batch_size=64, num_workers=2)
        
        # 评估模型
        test_loss, accuracy = test(model, device, test_loader)
    except Exception as e:
        print(f"运行中出错: {e}")
        print("请检查MNIST数据集是否正确加载，或者预训练权重是否正确加载")

if __name__ == '__main__':
    main()