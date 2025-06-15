# 深度学习模型代码实现笔记  

## 一、`train_alex.py`：AlexNet图像分类训练  
### （一）核心功能  
基于AlexNet架构实现自定义数据集的图像分类训练，包含数据加载、模型定义、训练与测试流程。  

### （二）关键技术点  
1. **数据集处理**  
   - 自定义`ImageTxtDataset`通过文本文件加载图像路径与标签  
   - 预处理：Resize至224×224、随机水平翻转、归一化（ImageNet均值/标准差）  
   ```python
   transform = transforms.Compose([
       transforms.Resize(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
   ```  

2. **模型结构**  
   - 简化版AlexNet：5卷积层+3全连接层，输出10分类  
   - 下采样采用MaxPool2d，输入3通道RGB图像  

3. **训练配置**  
   - 批量大小64，交叉熵损失+SGD优化器（lr=0.01，动量0.9）  
   - 每500步记录TensorBoard损失，epoch结束后测试集评估并保存模型  

4. **测试流程**  
   - `torch.no_grad()`禁用梯度，计算总损失与准确率  


### （三）学习要点  
- 自定义数据集加载与预处理流程  
- AlexNet卷积-池化-全连接层的架构设计  
- PyTorch训练循环（损失计算、优化器更新、模型评估）  
- 数据增强（随机翻转）对泛化能力的提升  


## 二、`transformer.py`：Vision Transformer实现  
### （一）核心功能  
基于Transformer架构实现Vision Transformer（ViT）模型，处理序列化图像数据用于分类。  


### （二）关键模块  
1. **核心组件**  
   - **FeedForward**：线性层+GELU激活+LayerNorm，实现特征变换  
   - **Attention**：多头自注意力机制，通过Softmax计算权重  
   - **Transformer层**：注意力模块+前馈模块，搭配残差连接  

2. **ViT模型结构**  
   - 图像序列化：将图像分割为patches，转换为序列输入  
   - 位置嵌入与类别嵌入（cls token）  
   - Transformer处理后通过全连接层输出分类结果  
   ```python
   # 关键结构示意
   class ViT(nn.Module):
       def forward(self, x):
           x = self.patch_embedding(x)  # 图像分块嵌入
           x = x + self.pos_embedding  # 位置嵌入
           x = self.transformer(x)  # Transformer处理
           return self.classifier(x[:, 0])  # 取cls token分类
   ```  

3. **测试验证**  
   - 输入随机张量`(batch, channels, height, width)`  
   - 输出logits形状`(batch, num_classes)`，验证模型有效性  


### （三）学习要点  
- Transformer核心机制：多头注意力、残差连接、LayerNorm  
- ViT将图像转为序列的处理逻辑，位置嵌入与cls token的作用  
- `einops`库张量操作（`rearrange`/`repeat`）简化维度变换  
- 序列化输入与分类输出的映射关系  


## 三、总结  
1. **CNN vs Transformer**  
   - AlexNet代表传统CNN架构，通过卷积核提取局部特征  
   - ViT展示Transformer在视觉任务的应用，通过序列建模捕获全局关系  

2. **关键技能**  
   - 自定义数据集处理流程（文本加载、预处理、增强）  
   - 模型训练全流程（损失函数、优化器、评估、可视化）  
   - 不同架构的核心设计思想与代码实现差异