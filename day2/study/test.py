import torch
import torchvision
from PIL import Image
from model import *
from torchvision import transforms

image_path = "./Image/img.png"
image = Image.open(image_path)
print(image)

image = image.convert('RGB')

# 改变成tensor格式
trans_reszie = transforms.Resize((32, 32))
trans_totensor = transforms.ToTensor()
transform = transforms.Compose([trans_reszie, trans_totensor])
image = transform(image)
print(image.shape)

# 加载训练模型
model = torch.load("model_save\\chen_9.pth",weights_only=False).to("cpu")

# print(model)

image = torch.reshape(image, (1, 3, 32, 32)).to("cpu")
# Image = Image.cuda()

# 将模型转换为测试模型
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

print(output.argmax(1))
