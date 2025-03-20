import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ 定义 MLP 模型
class MLP_MNIST(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平 28x28 -> 784
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)  # 输出 logits，不加 softmax
        return x

# 2️⃣ 数据加载（MNIST）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

train_dataset = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 3️⃣ 设备选择（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4️⃣ 初始化模型
model = MLP_MNIST()
state_dict_path = "../data/state_dicts/mlp_mnist.pth"  # 预训练模型路径
try:
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"成功加载预训练权重: {state_dict_path}")
except FileNotFoundError:
    print(f"未找到预训练权重: {state_dict_path}，将使用随机初始化的模型。")
model = model.to(device)

# 5️⃣ 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 6️⃣ 训练模型
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
    
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 7️⃣ 评估模型（推理）
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 8️⃣ 显示部分测试样本及预测结果
def imshow(img):
    img = img * 0.3081 + 0.1307  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# 获取一批测试数据
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# 进行预测
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 显示前 5 张图片及其预测
imshow(torchvision.utils.make_grid(images[:5].cpu()))
print("GroundTruth:", " ".join(f"{labels[j].item()}" for j in range(5)))
print("Predicted:  ", " ".join(f"{predicted[j].item()}" for j in range(5)))

# 9️⃣ 保存模型
# torch.save(model.state_dict(), "mlp_mnist.pth")
# print("Model saved as mlp_mnist.pth")
