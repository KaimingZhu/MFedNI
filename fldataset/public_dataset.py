"""
@Encoding:      UTF-8
@File:          public_dataset.py

@Introduction:  A enum class for public dataset
@Author:        Kaiming Zhu
@Date:          2023/7/19 20:54
@Reference:     https://pytorch.org/vision/stable/datasets.html
"""

from enum import Enum


class PublicDataset(Enum):
    """
    the value of enum will be introduced as the name of it.
    """
    # http://yann.lecun.com/exdb/mnist/
    Mnist = "Mnist"
    # https://github.com/zalandoresearch/fashion-mnist
    FashionMnist = "FashionMnist"
    # https://www.cs.toronto.edu/~kriz/cifar.html
    Cifar10 = "Cifar10"
    # https://www.cs.toronto.edu/~kriz/cifar.html
    Cifar100 = "Cifar100"