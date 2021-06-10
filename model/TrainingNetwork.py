# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dataTransforms = {
    'train':
        transforms.Compose([
            transforms.RandomResizedCrop(42),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
    'test':
        transforms.Compose([
            transforms.CenterCrop(42),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
}

PATH = r"..\datasets"
imageDatasets = {
    x: datasets.ImageFolder(os.path.join(PATH, x), dataTransforms[x])
    for x in ['train', 'test']
}
dataLoader = {
    x: DataLoader(imageDatasets[x],
                  batch_size=32,
                  shuffle=True,
                  num_workers=4)
    for x in ['train', 'test']
}
datasetSizes = {x: len(imageDatasets[x]) for x in ['train', 'test']}
GPUAvailable = torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bn_x = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn_conv1 = nn.BatchNorm2d(32, momentum=0.5)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=4,
                               stride=1,
                               padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32, momentum=0.5)
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn_conv3 = nn.BatchNorm2d(64, momentum=0.5)
        self.fc1 = nn.Linear(in_features=5 * 5 * 64, out_features=2048)
        self.bn_fc1 = nn.BatchNorm1d(2048, momentum=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.bn_fc2 = nn.BatchNorm1d(1024, momentum=0.5)
        self.fc3 = nn.Linear(in_features=1024, out_features=7)

    def forward(self, x):
        x = self.bn_x(x)
        x = functional.max_pool2d(torch.relu(self.bn_conv1(self.conv1(x))),
                                  kernel_size=3,
                                  stride=2,
                                  ceil_mode=True)
        x = functional.max_pool2d(torch.relu(self.bn_conv2(self.conv2(x))),
                                  kernel_size=3,
                                  stride=2,
                                  ceil_mode=True)
        x = functional.max_pool2d(torch.relu(self.bn_conv3(self.conv3(x))),
                                  kernel_size=3,
                                  stride=2,
                                  ceil_mode=True)
        # view是把一个矩阵重新排列成不同维度但不改变元素的函数
        # 这里 - 1就是把后面的矩阵展成一维数组，以便后面线性变换层操作
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = functional.dropout(x, training=self.training, p=0.4)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = functional.dropout(x, training=self.training, p=0.4)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def trainModel(model, criterion, optimizer, numOfEpochs):
    since = time.time()
    bestModel = model.state_dict()
    bestAcc = 0.0
    for epoch in range(numOfEpochs):
        print('Epoch {}/{}'.format(epoch, numOfEpochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
            runningLoss = 0.0
            runningAcc = 0
            for data in dataLoader[phase]:
                inputs, labels = data
                if GPUAvailable:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, prediction = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                runningLoss += loss.item()
                runningAcc += torch.sum(prediction == labels)
            # 要注意这里不可以使用"/"作除法
            epochLoss = torch.true_divide(runningLoss, datasetSizes[phase])
            epochAcc = torch.true_divide(runningAcc, datasetSizes[phase])
            # 这里的操作是为了方便作图
            if phase == 'train':
                allTrainLoss.append(epochLoss.item())
                allTrainAcc.append(epochAcc.item())
            else:
                allTestLoss.append(epochLoss.item())
                allTestAcc.append(epochAcc.item())
            # 这里是将训练数据可视化
            plt.figure(dpi=500)
            plt.plot(allTestLoss)
            plt.plot(allTrainLoss)
            plt.legend(["Test Loss", "Train Loss"])
            plt.show()
            plt.figure(dpi=500)
            plt.plot(allTrainAcc)
            plt.plot(allTestAcc)
            plt.legend(["Train Acc", "Test Acc"])
            plt.show()
            plt.close()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epochLoss, epochAcc))
            # 寻找最佳正确率对应的模型
            if phase == 'test' and epochAcc > bestAcc:
                bestAcc = epochAcc
                bestModel = model.state_dict()
            print()
        # 获取从第一次训练到目前的总运行时间
        runTime = time.time() - since
        # 完成一次训练，进行保存和总结果显示
        print('Training complete in {:0f}min {:.0f}s'.format(
            runTime // 60, runTime % 60))
        print('Best test Acc: {:4f}'.format(bestAcc))
        model.load_state_dict(bestModel)
        torch.save(model, 'best_model.pkl')
        torch.save(model.state_dict(), 'model_params.pkl')


if __name__ == '__main__':
    model = Model()
    allTrainLoss = []
    allTestLoss = []
    allTrainAcc = []
    allTestAcc = []
    if GPUAvailable:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    trainModel(model, criterion, optimizer, numOfEpochs=100)
    print(allTestAcc)
    print(allTrainAcc)
    print(allTrainLoss)
    print(allTestLoss)
