"""
@Encoding:      UTF-8
@File:          lenet.py

@Introduction:  implementation of LeNet
@Author:        Kaiming Zhu
@Date:          2023/8/6 22:30
@Reference:     https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master
"""

from typing import Tuple

from torch import nn


class LeNet5(nn.Module):
    def __init__(self, input_channel_amount: int = 1, output_class_amount: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel_amount, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, output_class_amount)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

    @classmethod
    def instance(cls, input_shape: Tuple[int], output_shape: Tuple[int]) -> nn.Module:
        input_channel_amount, output_class_amount = input_shape[0], output_shape[0]
        return cls(
            input_channel_amount=input_channel_amount,
            output_class_amount=output_class_amount
        )