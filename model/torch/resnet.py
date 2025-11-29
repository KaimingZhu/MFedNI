"""
@Encoding:      UTF-8
@File:          resnet.py

@Introduction:  implementation of ResNet
@Author:        Kaiming Zhu
@Date:          2023/8/6 21:32
@Reference:     https://github.com/akamaster/pytorch_resnet_cifar10
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb

    def forward(self, x):
        return self.lamb(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, block_amounts, input_channel_amount: int = 3, output_class_amount: int = 10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(input_channel_amount, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, block_amounts[0], stride=1)
        self.layer2 = self._make_layer(block, 32, block_amounts[1], stride=2)
        self.layer3 = self._make_layer(block, 64, block_amounts[2], stride=2)
        self.linear = nn.Linear(64, output_class_amount)

    def _make_layer(self, block, planes, block_amounts, stride):
        strides = [stride] + [1] * (block_amounts-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @classmethod
    def resnet20_instance(cls, input_shape: Tuple[int], output_shape: Tuple[int]) -> nn.Module:
        input_channel_amount, output_class_amount = input_shape[0], output_shape[0]
        return cls(
            block=BasicBlock,
            block_amounts=[3, 3, 3],
            input_channel_amount=input_channel_amount,
            output_class_amount=output_class_amount
        )

    @classmethod
    def resnet32_instance(cls, input_shape: Tuple[int], output_shape: Tuple[int]) -> nn.Module:
        input_channel_amount, output_class_amount = input_shape[0], output_shape[0]
        return cls(
            block=BasicBlock,
            block_amounts=[5, 5, 5],
            input_channel_amount=input_channel_amount,
            output_class_amount=output_class_amount
        )

    @classmethod
    def resnet44_instance(cls, input_shape: Tuple[int], output_shape: Tuple[int]) -> nn.Module:
        input_channel_amount, output_class_amount = input_shape[0], output_shape[0]
        return cls(
            block=BasicBlock,
            block_amounts=[7, 7, 7],
            input_channel_amount=input_channel_amount,
            output_class_amount=output_class_amount
        )

    @classmethod
    def resnet56_instance(cls, input_shape: Tuple[int], output_shape: Tuple[int]) -> nn.Module:
        input_channel_amount, output_class_amount = input_shape[0], output_shape[0]
        return cls(
            block=BasicBlock,
            block_amounts=[9, 9, 9],
            input_channel_amount=input_channel_amount,
            output_class_amount=output_class_amount
        )

    @classmethod
    def resnet110_instance(cls, input_shape: Tuple[int], output_shape: Tuple[int]) -> nn.Module:
        input_channel_amount, output_class_amount = input_shape[0], output_shape[0]
        return cls(
            block=BasicBlock,
            block_amounts=[18, 18, 18],
            input_channel_amount=input_channel_amount,
            output_class_amount=output_class_amount
        )

    @classmethod
    def resnet1202_instance(cls, input_shape: Tuple[int], output_shape: Tuple[int]) -> nn.Module:
        input_channel_amount, output_class_amount = input_shape[0], output_shape[0]
        return cls(
            block=BasicBlock,
            block_amounts=[200, 200, 200],
            input_channel_amount=input_channel_amount,
            output_class_amount=output_class_amount
        )
