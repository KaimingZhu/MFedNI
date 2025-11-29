"""
@Encoding:      UTF-8
@File:          initializer.py

@Introduction:  initializer for torch model
@Author:        Kaiming Zhu
@Date:          2023/8/6 20:40
"""

from enum import Enum
from typing import Callable, Dict
import math

import torch
import torch.nn as nn


class Initializer(Enum):
    KaimingNormal = 1,
    KaimingUniform = 2,
    XavierNormal = 3,
    XavierUniform = 4,
    LecunNormal = 5,
    LecunUniform = 6,
    RandomNormal = 7,
    RandomUniform = 8

    def apply(self, model: nn.Module):
        initializer_by_enum: Dict[Initializer, Callable] = {
            Initializer.KaimingNormal: nn.init.kaiming_normal_,
            Initializer.KaimingUniform: nn.init.kaiming_uniform_,
            Initializer.XavierUniform: nn.init.xavier_uniform_,
            Initializer.XavierNormal: nn.init.xavier_normal_,
            Initializer.LecunNormal: Initializer.lecun_normal_,
            Initializer.LecunUniform: Initializer.lecun_uniform_,
            Initializer.RandomNormal: nn.init.normal_,
            Initializer.RandomUniform: nn.init.uniform_,
        }

        initializer = initializer_by_enum[self]
        if initializer is not None:
            def _weight_init(layer: nn.Module):
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    initializer(layer.weight)

            model.apply(_weight_init)

    @staticmethod
    def lecun_normal_(tensor: torch.Tensor):
        """
        References:
            https://discuss.pytorch.org/t/how-can-i-apply-lecun-weight-initialization-for-my-linear-layer/115418
        """
        input_size = tensor.shape[-1]  # Assuming that the weights' input dimension is the last.
        std = math.sqrt(1 / input_size)
        with torch.no_grad():
            tensor.normal_(-std, std)

    @staticmethod
    def lecun_uniform_(tensor: torch.Tensor):
        """
        References:
            https://github.com/bjkomer/pytorch-legendre-memory-unit/blob/master/lmu.py
            https://github.com/deepsound-project/samplernn-pytorch/blob/master/nn.py#L46
        """
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
