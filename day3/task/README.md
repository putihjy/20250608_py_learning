# 深度学习模型与数据处理笔记  

## 1. 数据集处理  
### 1.1 数据集划分  
- **划分方式**：使用`train_test_split`按比例划分，确保数据分布均匀。  
```python
from sklearn.model_selection import train_test_split
train_images, val_images = train_test_split(images, train_size=0.7, random_state=42)
```  

### 1.2 数据集加载与预处理  
- **自定义数据集**：通过`.txt`文件加载图片路径和标签。  
```python
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, folder_name, transform):
        self.transform = transform
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.imgs_path = [line.split()[0] for line in lines]
        self.labels = [int(line.split()[1].strip()) for line in lines]
```  
- **预处理**：调整尺寸、归一化。  
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```  

## 2. 神经网络模型  
### 2.1 GoogLeNet  
- **Inception模块**：多分支卷积（1×1、3×3、5×5）+池化，增强特征提取。  
```python
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_features):
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        # 其他分支结构类似
```  

### 2.2 MobileNetV2  
- **Inverted Residual模块**：先扩维再深度可分离卷积，减少计算量。  
```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = int(round(in_channels * expand_ratio))
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6()
        )
```  

### 2.3 ResNet18  
- **残差结构**：通过跳跃连接解决深层网络梯度消失问题。  
```python
from torchvision.models import resnet18
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # 替换分类层
```  

## 3. 模型训练与测试  
### 3.1 训练配置  
- **损失函数**：交叉熵损失。  
```python
criterion = nn.CrossEntropyLoss()
```  
- **优化器**：Adam动态调整学习率。  
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```  

### 3.2 测试与评估  
- **准确率计算**：  
```python
correct = 0
with torch.no_grad():
    for imgs, labels in dataloader:
        outputs = model(imgs)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
accuracy = correct / len(dataloader.dataset)
```  
- **日志记录**：TensorBoard可视化。  
```python
writer = SummaryWriter("logs")
writer.add_scalar("Train Loss", loss, epoch)
writer.add_scalar("Test Acc", accuracy, epoch)
```  

## 4. 激活函数与数据可视化  
### 4.1 ReLU激活函数  
- **特点**：非线性、加速收敛、缓解梯度消失。  
```python
self.relu = nn.ReLU()  # 应用于卷积层后
```  

### 4.2 TensorBoard可视化  
- **输入输出对比**：  
```python
writer.add_images("Input Images", imgs, step)
writer.add_images("Activation Output", output, step)
```  

## 5. 数据准备脚本  
### 5.1 创建txt标签文件  
- **功能**：自动生成图片路径与标签映射文件。  
```python
def create_txt_file(root_dir, txt_path):
    with open(txt_path, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            cat_path = os.path.join(root_dir, category)
            for img_name in os.listdir(cat_path):
                f.write(f"{os.path.join(cat_path, img_name)} {label}\n")
```  
- **调用示例**：  
```python
create_txt_file("data/train", "train.txt")
create_txt_file("data/val", "val.txt")
```