import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                               目录                                 #
# ================================================================== #

# 1. 自动求导示例 1               (Line 25 to 39)
# 2. 自动求导示例 2               (Line 46 to 83)
# 3. 从Numpy载入数据                (Line 90 to 97)
# 4. 输入                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189)


# ================================================================== #
#                     1. 自动求导示例 1                               #
# ================================================================== #

# 创建张量.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 创建计算图
y = w * x + b    # y = 2 * x + 3

# 计算梯度
y.backward()

# 输出梯度
print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1


# ================================================================== #
#                    2. 自动求导示例 2                                #
# ================================================================== #

# 创建形状为(10, 3)和(10, 2)的张量
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 创建全连接层
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# 创建损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# 前向传播
pred = linear(x)

# 计算损失
loss = criterion(pred, y)
print('loss: ', loss.item())

# 反向传播
loss.backward()

# 输出梯度
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

# 梯度下降
optimizer.step()

# 也可以在底层进行梯度下降
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# 梯度下降后的输出
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# 创建Numpy数组
x = np.array([[1, 2], [3, 4]])

# Numpy数组转为张量
y = torch.from_numpy(x)

# 张量转为Numpy数组
z = y.numpy()


# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #

# 下载并构建CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# 从硬盘读取数据
image, label = train_dataset[0]
print (image.size())
print (label)

# 数据载入器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# 当迭代开始，队列和线程开始从文件载入数据
data_iter = iter(train_loader)

# 小批量图像和标签
images, labels = data_iter.next()

# 数据载入器的使用如下
for images, labels in train_loader:
    # 训练代码.
    pass


# ================================================================== #
#                5. Input pipeline for custom dataset                #
# ================================================================== #

# 按如下方式构建自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. 初始化文件路径或者文件名
        pass
    def __getitem__(self, index):
        # TODO
        # 1. 从文件读取数据 (使用numpy.fromfile, PIL.Image.open).
        # 2. 预处理数据 (使用torchvision.Transform).
        # 3. 返回数据(图像和标签).
        pass
    def __len__(self):
        # 将0改为数据集的大小.
        return 0

# 使用预创建的数据载入器.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=False)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# 下载并载入ResNet-18模型.
resnet = torchvision.models.resnet18(pretrained=True)

# 如果只需要微调或者迁移学习模型的高层
for param in resnet.parameters():
    param.requires_grad = False

# 替换高层用于微调模型或迁移学习
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# 前向传播
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# 保存和加载模型
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# 只保存和加载模型的参数(推荐)
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))