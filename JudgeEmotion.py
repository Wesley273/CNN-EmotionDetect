# -*- coding: utf-8 -*-
from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms


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


def judgeEmotion(inputs):
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    # 这时input格式为[C,H,W]，C为图片通道，H W代表图片为H*W大小的图片
    inputs = trans(inputs)
    # 增加一维，输出的img格式为[1,C,H,W]
    inputs = inputs.unsqueeze(0)
    # 找到输出中可信度最大的，“_,”是一个临时变量，只是为了让它接受多余的参数
    _, predicted = torch.max(model(inputs), 1)
    probability = torch.nn.functional.softmax(
        (model(inputs)), dim=1).detach().numpy().flatten()
    emotion = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprised',
        6: 'normal'
    }
    return emotion[predicted.item()], probability


def getBarChart(probability, coordinate):
    emotion = {
        0: 'angry:',
        1: 'disgust:',
        2: 'fear:',
        3: 'happy:',
        4: 'sad:',
        5: 'surprised:',
        6: 'normal:'
    }
    x = coordinate[0]
    y = coordinate[1]
    w = coordinate[2]
    h = coordinate[3]
    for index, item in enumerate(probability):
        str_probability = str(item * 100)
        i = index + 1
        cv2.putText(frame, emotion[index], (x, y + h + 15 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, 3)
        cv2.rectangle(frame, (x + 60, y + h + 15 * i - 5),
                      (x + int(60 + (w - 60) * item), y + h + 15 * i),
                      (255, 0, 0), -2)
        cv2.putText(frame, str_probability[0:4] + '%',
                    (x + int(60 + (w - 60) * item) + 5, y + h + 15 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, 3)


if __name__ == '__main__':
    # 下面导入已经训练好的模型
    model = Model()
    model.load_state_dict(torch.load(r'.\model\model_params.pkl'))
    model.eval()
    torch.no_grad()
    face_cascade = cv2.CascadeClassifier(
        r".\datasets\haarcascade_frontalface_default.xml")
    # 设置摄像头 0是默认的摄像头 如果有多个摄像头的话，可以设置1,2,3....
    cap = cv2.VideoCapture(0)
    # 进入无限循环
    while True:
        # 将摄像头拍到的图像作为frame值
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0),
                                  2)
            # 图片裁切，得到人脸
            face = cv2.resize(frame[y:(y + h), x:(x + w)], (42, 42))
            emotion, probability = judgeEmotion(Image.fromarray(face))
            cv2.putText(frame, emotion, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, 4)
            getBarChart(probability, (x, y, w, h))
        # 将frame的值显示出来 有两个参数 前一个是窗口名字，后面是值
        cv2.imshow('Camera', frame)
        # 判断退出的条件 当按下'Q'键的时候呢，就退出
        c = cv2.waitKey(1)
        if c == ord('Q'):
            break
    cap.release()  # 常规操作
    cv2.destroyAllWindows()
