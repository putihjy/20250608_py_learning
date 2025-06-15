### 深度学习与卷积神经网络基础  
#### 一、深度学习概述  
- **定义**：机器学习分支，通过多层神经网络学习数据复杂模式与特征。  
- **应用领域**：图像识别、语音识别、自然语言处理、推荐系统等。  

#### 二、神经网络基础  
- **神经元模型**：模拟生物神经元，输入经加权求和与激活函数处理后输出。  
- **激活函数**：引入非线性，常见Sigmoid、ReLU、Tanh等。  
- **损失函数**：衡量预测与真实值差异，如均方误差（MSE）、交叉熵损失。  
- **优化算法**：梯度下降及其变体（SGD、Adam等）调整网络权重。  

#### 三、卷积神经网络（CNN）  
##### （一）卷积操作  
- **卷积核**：提取输入数据局部特征。  
- **步长（Stride）**：卷积核滑动步长。  
- **填充（Padding）**：边缘补像素保持输出尺寸。  
```python
import torch
input = torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]]).reshape(1,1,5,5)
kernel = torch.tensor([[1,2,1],[0,1,0],[2,1,0]]).reshape(1,1,3,3)
output = torch.nn.functional.conv2d(input, kernel, stride=1)
print(output)
```  

##### （二）网络结构  
- **卷积层**：提取特征。  
- **池化层**：降维减计算量，如最大池化、平均池化。  
- **全连接层**：展平特征后分类/回归。  
```python
import torch.nn as nn
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2), nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2), nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(1024,64), nn.Linear(64,10)
        )
    def forward(self,x): return self.model(x)
model = CNNModel()
input = torch.ones((64,3,32,32))
print(model(input).shape)
```  

#### 四、模型训练与测试  
##### （一）数据集准备  
- **CIFAR10**：6万张32×32彩色图，10类。  
- **数据加载**：  
```python
import torchvision
train_data = torchvision.datasets.CIFAR10(".", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(".", train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
```  

##### （二）模型训练  
- **配置**：交叉熵损失 + SGD优化器。  
- **训练流程**：前向计算损失→反向传播更新权重。  
```python
import torch.optim as optim
model = CNNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10):
    for imgs, targets in train_loader:
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if total_train_step % 500 == 0: print(f"训练loss: {loss.item()}")
```  

##### （三）模型测试与保存  
- **评估**：计算测试集准确率。  
- **保存**：`torch.save(model, "model.pth")`。  
```python
total_test_loss, total_accuracy = 0.0, 0
model.eval()
with torch.no_grad():
    for imgs, targets in test_loader:
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_test_loss += loss.item()
        total_accuracy += (outputs.argmax(1) == targets).sum()
print(f"测试loss: {total_test_loss}, 准确率: {total_accuracy / len(test_data)}")
```  

#### 五、总结  
- 掌握神经网络基础（神经元、激活函数等）。  
- 理解CNN结构与卷积操作，能构建训练模型。  
- 熟悉数据集处理、模型训练测试及可视化流程。