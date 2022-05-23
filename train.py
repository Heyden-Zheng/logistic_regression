import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import LogisticRegression

'''
步骤：
    1.模型参数设置
    2.加载数据集
    3.编写逻辑回归模型
    4.训练模型
    5.测试模型效果
'''
# 图片大小
input_size = 784  # 每张图片像素为28*28,因此每张图片会产生784个数据,代表样本特征
# 加载批训练的数据个数
num_classes = 10  # 样本是0-9的手写数字,共10类
num_epochs = 10  # 训练10轮
batch_size = 50
# 设置超参数
learning_rate = 0.001  # 学习率

# 2.数据集
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 加载数据集
#   参数说明：
#   dataset:要加载的数据集；batch_size：加载批训练的数据个数；shuffle：若为True，则每个epoch都重新排列数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 模型初始化
model = LogisticRegression(input_size, num_classes)
loss = nn.CrossEntropyLoss()  # 最小二乘Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化函数


# 训练模型
class train():
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 将加载的数据转为Tensor
            images = Variable(images.view(-1, 28 * 28))  # images.view(-1, 28 * 28)表示将数据转成二维,行数为batch_size,列数为28*28
            labels = Variable(labels)
            # 设置梯度为零
            optimizer.zero_grad()
            # 更新参数
            outputs = model(images)
            train_loss = loss(outputs, labels)  # 计算梯度
            train_loss.backward()
            optimizer.step()

            # 每隔100次打印一次结果
            if (i + 1) % 100 == 0:
                print('Epoch:[%d/%d], Step:[%d/%d], Loss:%.4f' % (epoch + 1, num_epochs,
                                                                  i + 1,
                                                                  len(train_dataset) // batch_size,
                                                                  train_loss.item()))


# 测试模型，计算模型的精度
class test():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = model(images)  # outputs的大小为batch_size*num_classes,即50*10,代表该批次已分好类(10类)
        _, predicted = torch.max(outputs.data, dim=1)  # '_'表示返回一行中最大的数,即最有可能的那个类的value值;predicted表示value所在的位置
        total += labels.size(0)  # 累加每批次的样本数,获得一个epoch所有的样本数
        correct += (predicted == labels).sum()  # 累加每批次预测正确的样本数,获得一个epoch所有预测正确的样本数
        # 打印模型的预测准确率
    print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
    torch.save(model.state_dict(), 'model.pkl')


train()
test()
