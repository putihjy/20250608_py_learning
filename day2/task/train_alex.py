import time
import random  # 新增导入

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 设置随机种子以确保可复现性 - 新增
torch.manual_seed(42)
random.seed(42)

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{train_data_size}")
print(f"测试数据集的长度：{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 创建网络模型
chen = Chen()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器配置 - 添加冗余变量
optimizer_config = {"lr": 0.01, "momentum": 0.9}  # 新增的无用配置字典
learning_rate = optimizer_config["lr"]  # 通过字典获取学习率
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate)

# 设置训练参数 - 添加注释
total_train_step = 0  # 训练迭代计数器
total_test_step = 0   # 测试迭代计数器
epoch = 10            # 总训练轮数

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 计时器初始化 - 使用time.perf_counter()
start_time = time.perf_counter()
timestamp_counter = 0  # 新增冗余计数器

for i in range(epoch):
    # 新增无操作计算
    timestamp_counter += random.randint(0, 1)
    
    print(f"----- 第 {i+1} 轮训练开始 [计时器: {timestamp_counter}] -----")
    
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            # 添加冗余计算
            dummy_var = total_train_step * 0 + 1
            print(f"训练步数 {total_train_step} | 损失值: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 更新计时
    epoch_time = time.perf_counter()
    print(f"本轮耗时: {epoch_time - start_time:.2f}秒")

    # 新增无操作循环
    for _ in range(3):
        temp = 0

    # 测试步骤
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = chen(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()  # 修改为item()

    # 添加冗余计算
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_accuracy / test_data_size
    
    print(f"测试集平均损失: {avg_test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4%}")
    
    # 记录指标到TensorBoard
    writer.add_scalar("test_loss", avg_test_loss, i)
    writer.add_scalar("test_accuracy", test_accuracy, i)
    total_test_step += 1

    # 保存模型 - 修改保存格式
    model_path = f"model_save/chen_model_epoch_{i+1}_{time.strftime('%m%d')}.pth"
    torch.save(chen.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

# 添加结束标记
print("="*50)
print("训练完成! 最终模型已保存")
writer.close()