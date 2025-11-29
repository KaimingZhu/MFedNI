"""
@Encoding:      UTF-8
@File:          modal_merge_model.py

@Introduction:  implementation of ModalMergeModel
@Author:        Kaiming Zhu
@Date:          2023/12/21 6:03
@Reference:     Songcan Yu, et al. "Robust multimodal federated learning for incomplete modalities." Computer Communications (2023).
"""

from functools import reduce
from operator import mul
from typing import Tuple

import torch.nn as nn


class ModalMergeModel(nn.Module):
    """ ModalMergeModel for FedUnion, FedProx and e.t.c.
    Notes:
        We defined `uniform_layers` with `__set_attr__` and `__get_attr__`, rather than defined and save them into list.
        That is because, parameters in list-saved layers will fail to move to GPU when we use `model.cuda()`. Please
        refer to the linkage below for more details.

    See Also:
        https://discuss.pytorch.org/t/some-tensors-getting-left-on-cpu-despite-calling-model-to-cuda/112915/7
    """
    def __init__(self, *input_shapes, output_shape):
        super().__init__()
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)

        # Add uniform layers w.r.t. amount of modal
        for index, input_shape in enumerate(input_shapes):
            # Equals to: 1 * elem0 * elem1 * ... * elem_n
            dimension = reduce(mul, list(input_shape) + [1])
            layer = nn.Sequential(
                nn.Linear(in_features=dimension, out_features=24),
                nn.Sigmoid()
            )
            self.__setattr__(f"uniform_layer{index}", layer)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=24, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=output_shape[0])
        )

    def forward(self, *modal_datas, need_extract_feat=False):
        flatten_modal_datas = [self.flatten_layer(modal_data) for modal_data in modal_datas]
        uniform_modal_datas = []
        for index, modal_data in enumerate(flatten_modal_datas):
            layer: nn.Module = self.__getattr__(f"uniform_layer{index}")
            uniform_modal_datas.append(layer(modal_data))

        merged_feature = sum(uniform_modal_datas)
        output = self.classifier(merged_feature)

        if need_extract_feat:
            return merged_feature, output
        else:
            return output

    @classmethod
    def instance(cls, *input_shapes: Tuple[Tuple[int]], output_shape: Tuple[int]) -> nn.Module:
        input_shapes = list(input_shapes)
        return cls(*input_shapes, output_shape=output_shape)
