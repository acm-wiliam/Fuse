import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 设置随机种子以确保结果可重复
torch.manual_seed(0)
np.random.seed(0)

# CIFAR-10 数据集的类别名称
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# 定义3x3卷积，带有padding
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding
    
    参数:
    in_planes (int): 输入通道数
    out_planes (int): 输出通道数
    stride (int): 卷积步长
    groups (int): 分组卷积的组数
    dilation (int): 卷积核膨胀率
    
    返回:
    nn.Conv2d: 3x3卷积层
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,  # 3x3卷积核
        stride=stride,  # 步长
        padding=dilation,  # 填充，保持特征图大小
        groups=groups,  # 分组数，默认为1表示常规卷积
        bias=False,  # 不使用偏置
        dilation=dilation,  # 卷积核膨胀率，默认为1表示常规卷积
    )

# 定义1x1卷积
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution
    
    参数:
    in_planes (int): 输入通道数
    out_planes (int): 输出通道数
    stride (int): 卷积步长
    
    返回:
    nn.Conv2d: 1x1卷积层，通常用于改变通道数
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义ResNet的基本块
class BasicBlock(nn.Module):
    # 通道扩张因子，BasicBlock不改变通道数
    expansion = 1

    def __init__(
        self,
        inplanes,  # 输入通道数
        planes,  # 输出通道数（实际输出为planes * expansion）
        stride=1,  # 步长，控制特征图大小
        downsample=None,  # 下采样层，用于调整输入和输出维度匹配
        groups=1,  # 分组卷积的组数
        base_width=64,  # 基础宽度
        dilation=1,  # 卷积核膨胀率
        norm_layer=None,  # 归一化层类型
    ):
        super(BasicBlock, self).__init__()
        # 如果未指定归一化层，则使用BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # BasicBlock只支持常规卷积
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        # 第一个卷积层，当stride>1时降低特征图大小
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        # 第二个卷积层，保持特征图大小
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample  # 下采样层
        self.stride = stride  # 步长

    def forward(self, x):
        identity = x  # 保存输入，用于残差连接

        # 第一个卷积块：卷积+批归一化+ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积块：卷积+批归一化（注意没有ReLU）
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样（用于匹配输入和输出维度）
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：将输入与卷积输出相加
        out += identity
        out = self.relu(out)  # 相加后再应用ReLU

        return out

# ResNet网络架构
class ResNet(nn.Module):
    def __init__(
        self,
        block,  # 基本块类型（BasicBlock或Bottleneck）
        layers,  # 每层中基本块的数量，例如[2,2,2,2]表示4个层各有2个基本块
        num_classes=10,  # 类别数
        zero_init_residual=False,  # 是否将残差分支的最后一个BN初始化为0
        groups=1,  # 分组卷积的组数
        width_per_group=64,  # 每组的宽度
        replace_stride_with_dilation=None,  # 是否用膨胀卷积替代步长
        norm_layer=None,  # 归一化层类型
    ):
        super(ResNet, self).__init__()
        # 如果未指定归一化层，则使用BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # 初始化参数
        self.inplanes = 64  # 初始通道数
        self.dilation = 1  # 初始膨胀率
        
        # 是否用膨胀卷积替代步长
        if replace_stride_with_dilation is None:
            # 对应3个layer（layer2,3,4）
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: 原ImageNet使用kernel_size=7,stride=2,padding=3
        # 对CIFAR10的修改：使用更小的卷积核(3x3)和步长(1)，适应32x32的小图像
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.bn1 = norm_layer(self.inplanes)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化，减小特征图
        
        # 四个层，每个层包含多个基本块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，输出类别数

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用kaiming正态分布初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 零初始化残差分支中的最后一个BN层
        # 这样每个残差块在初始状态下相当于恒等映射
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """创建一个包含多个基本块的层
        
        参数:
        block: 基本块类型
        planes: 基本块的输出通道数
        blocks: 基本块的数量
        stride: 第一个基本块的步长
        dilate: 是否使用膨胀卷积
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 如果使用膨胀卷积，则增加膨胀率而不是使用步长
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # 当步长不为1或输入输出通道数不匹配时，需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # 1x1卷积用于调整通道数和特征图大小
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 添加第一个基本块（可能有下采样）
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        # 更新输入通道数为当前层的输出通道数
        self.inplanes = planes * block.expansion
        
        # 添加剩余的基本块（无下采样）
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        # 将所有基本块组合成一个Sequential模块
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = self.avgpool(x)
        # 将特征图展平
        x = x.reshape(x.size(0), -1)
        # 全连接层分类
        x = self.fc(x)

        return x

# 创建ResNet18模型（预训练或非预训练）
def resnet18(pretrained=False, device="cpu"):
    """构建ResNet-18模型
    
    参数:
    pretrained (bool): 如果为True，返回预训练模型
    device (str): 模型所在设备，'cpu'或'cuda'
    
    返回:
    ResNet: ResNet-18模型实例
    """
    # ResNet18使用BasicBlock块，每层分别包含[2,2,2,2]个块
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    
    # 如果使用预训练模型
    if pretrained:
        # 尝试从不同可能的路径加载预训练权重
        try:
            # 先尝试从Fuse/data/state_dicts目录加载
            script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'state_dicts'))
            state_dict_path = os.path.join(script_dir, "resnet18.pt")
            
            # 如果不存在，尝试从PyTorch_CIFAR10目录加载
            if not os.path.exists(state_dict_path):
                script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PyTorch_CIFAR10'))
                state_dict_path = os.path.join(script_dir, "cifar10_models", "state_dicts", "resnet18.pt")
            
            # 加载预训练权重
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"成功加载预训练权重: {state_dict_path}")
        except Exception as e:
            print(f"无法加载预训练权重: {e}")
            print("尝试从多个路径加载权重失败，将使用随机初始化的模型")
    
    return model

# 数据预处理的均值和标准差（与原始模型训练时使用的相同）
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)

def prepare_data(batch_size=64, num_workers=4):
    """准备CIFAR-10数据集的数据加载器
    
    参数:
    batch_size (int): 批量大小
    num_workers (int): 数据加载的工作线程数
    
    返回:
    DataLoader: 测试数据加载器
    """
    # 测试数据变换（与训练时相同）
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    # 尝试不同的数据路径，优先使用已下载的数据集
    data_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),  # Fuse/data
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'),  # 上级目录的data
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'PyTorch_CIFAR10', 'data')  # PyTorch_CIFAR10/data
    ]
    
    # 尝试每个路径直到成功
    test_dataset = None
    for data_path in data_paths:
        try:
            print(f"尝试从路径加载数据: {data_path}")
            test_dataset = CIFAR10(
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
            print("尝试下载CIFAR10数据集...")
            test_dataset = CIFAR10(
                root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
                train=False,
                download=True,
                transform=transform_test
            )
            print("成功下载CIFAR10测试集")
        except Exception as e:
            raise RuntimeError(f"无法加载或下载CIFAR10数据集: {e}")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 不打乱数据顺序
        num_workers=num_workers,  # 数据加载的工作线程数
        pin_memory=True,  # 将数据加载到CUDA固定内存，加速数据传输到GPU
    )
    
    return test_loader

def denormalize(tensor, mean=CIFAR_MEAN, std=CIFAR_STD):
    """将归一化的图像张量转换回原始值范围
    
    参数:
    tensor (Tensor): 归一化的图像张量
    mean (tuple): 用于归一化的均值
    std (tuple): 用于归一化的标准差
    
    返回:
    Tensor: 反归一化后的图像张量
    """
    # 克隆张量避免原地修改
    tensor = tensor.clone()
    
    # 反归一化每个通道
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # 限制值到[0, 1]范围
    return torch.clamp(tensor, 0, 1)

def visualize_results(model, test_loader, device, num_samples=5):
    """可视化模型预测结果
    
    参数:
    model (nn.Module): 待评估的模型
    test_loader (DataLoader): 测试数据加载器
    device (torch.device): 用于推理的设备
    num_samples (int): 要显示的样本数量
    """
    # 设置模型为评估模式
    model.eval()
    
    # 获取一批次数据
    images, labels = next(iter(test_loader))
    
    # 限制样本数量
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 将图像送入模型进行推理
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 将图像从GPU复制回CPU并反归一化
    images = images.cpu()
    denorm_images = denormalize(images)
    
    # 可视化结果
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        # 将张量转换为numpy数组并调整通道顺序
        img = denorm_images[i].permute(1, 2, 0).numpy()
        
        # 显示图像
        axes[i].imshow(img)
        
        # 设置标题：预测类别 (真实类别)
        title_color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(
            f'Pred: {CIFAR10_CLASSES[predicted[i]]}\nTrue: {CIFAR10_CLASSES[labels[i]]}',
            color=title_color
        )
        
        # 移除坐标轴
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('resnet18_predictions.png')
    plt.show()

def inference_on_single_image(model, image_path, device):
    """对单个图像进行推理
    
    参数:
    model (nn.Module): 待评估的模型
    image_path (str): 图像路径
    device (torch.device): 用于推理的设备
    
    返回:
    int: 预测的类别
    float: 预测的置信度
    """
    # 设置模型为评估模式
    model.eval()
    
    # 加载并预处理图像
    transform = T.Compose([
        T.Resize((32, 32)),  # 调整大小为CIFAR-10图像尺寸
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # 添加批次维度
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction_idx = torch.argmax(probabilities).item()
        confidence = probabilities[prediction_idx].item()
    
    return prediction_idx, confidence, image

def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的性能
    
    参数:
    model (nn.Module): 待评估的模型
    test_loader (DataLoader): 测试数据加载器
    device (torch.device): 用于评估的设备
    
    返回:
    float: 测试准确率
    """
    # 设置模型为评估模式
    model.eval()
    
    correct = 0
    total = 0
    
    # 在不计算梯度的情况下进行推理
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy

def extract_cifar10():
    """尝试手动解压CIFAR10数据集
    
    返回:
    bool: 解压是否成功
    """
    import tarfile
    
    # 可能的tar.gz文件位置
    tar_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cifar-10-python.tar.gz'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cifar-10-python.tar.gz')
    ]
    
    # 尝试解压每个可能的路径
    for tar_path in tar_paths:
        if os.path.exists(tar_path):
            try:
                print(f"尝试解压文件: {tar_path}")
                data_dir = os.path.dirname(tar_path)
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=data_dir)
                print(f"成功解压CIFAR10数据集到: {data_dir}")
                return True
            except Exception as e:
                print(f"解压{tar_path}时出错: {e}")
    
    print("未找到CIFAR10压缩文件或解压失败")
    return False

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 首先尝试解压数据集
    extract_cifar10()
    
    # 创建模型
    model = resnet18(pretrained=True, device=device)
    model = model.to(device)
    
    try:
        # 准备数据
        test_loader = prepare_data(batch_size=64, num_workers=2)
        
        # 评估模型
        evaluate_model(model, test_loader, device)
        
        # 可视化结果
        visualize_results(model, test_loader, device, num_samples=5)
    except Exception as e:
        print(f"运行中出错: {e}")
        print("请检查CIFAR10数据集是否正确解压，或者预训练权重是否正确加载")
    
    # 如果有特定图像要测试，可以使用以下代码
    # image_path = "path/to/your/image.jpg"
    # if os.path.exists(image_path):
    #     pred_idx, confidence, image = inference_on_single_image(model, image_path, device)
    #     print(f"预测类别: {CIFAR10_CLASSES[pred_idx]}, 置信度: {confidence:.4f}")
    #     
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(image)
    #     plt.title(f"预测: {CIFAR10_CLASSES[pred_idx]} ({confidence:.4f})")
    #     plt.axis('off')
    #     plt.show()

if __name__ == "__main__":
    main()