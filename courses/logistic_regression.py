import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# 超参数
input_size = 28 * 28  # 784
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST数据集 (图像和标签)
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# 数据载入器 (输入流程)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 逻辑回归模型
model = nn.Linear(input_size, num_classes)

# 损失函数和优化器
# nn.CrossEntropyLoss() 计算内部softmax
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 改变图像形状 (batch_size, input_size)
        images = images.reshape(-1, input_size)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播及优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 测试模型
# 不计算梯度 (为了提高内存效率)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# 保存模型
torch.save(model.state_dict(), 'model.ckpt')