"""
@Encoding:      UTF-8
@File:          modal_concat_model.py

@Introduction:  Definition of ModalConcatModel
@Author:        Kaiming Zhu
@Date:          2023/12/25 22:24
@Reference:     'baseline_autofed_pytorch.py' by S. Yu
"""

from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn

class ModalConcatModel(nn.Module):
    def __init__(self, *input_shapes, output_shape):
        super().__init__()
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)

        # Classifier
        # For each modal, dimension = 1 * shape[0] * shape[1] * ... * shape[n]
        dimensions = [reduce(mul, [1] + list(shape)) for shape in input_shapes]
        dimension = sum(dimensions)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=dimension, out_features=64, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=output_shape[0], bias=True)
        )

    def forward(self, *modal_datas, need_extract_feat=False):
        flatten_modal_datas = [self.flatten_layer(modal_data) for modal_data in modal_datas]
        concat_feature = torch.cat(tensors=flatten_modal_datas, axis=1)
        output = self.classifier(concat_feature)

        if need_extract_feat:
            return concat_feature, output
        else:
            return output

    @classmethod
    def instance(cls, *input_shapes: Tuple[Tuple[int]], output_shape: Tuple[int]) -> nn.Module:
        input_shapes = list(input_shapes)
        return cls(*input_shapes, output_shape=output_shape)