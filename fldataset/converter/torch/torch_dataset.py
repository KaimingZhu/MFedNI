"""
@Encoding:      UTF-8
@File:          torch_dataset.py

@Introduction:  Definition of `TorchDataset`
@Author:        Kaiming Zhu
@Date:          2023/8/16 3:52
@Reference:     https://saturncloud.io/blog/how-to-apply-a-custom-transform-to-your-custom-dataset-in-pytorch/
"""

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import fldataset


class TorchDataset(Dataset):
    def __init__(
        self,
        dataset: fldataset.Dataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.transform: Optional[Callable] = transform
        self.target_transform: Optional[Callable] = target_transform

    @property
    def datas(self) -> np.ndarray:
        return self.dataset.datas

    @property
    def targets(self) -> np.ndarray:
        return self.dataset.labels

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        data = self.datas[index]
        target = self.targets[index]

        if self.transform is not None:
            tensor_data = self.transform(data)
        else:
            tensor_data = torch.from_numpy(np.asarray(data))

        if self.target_transform is not None:
            tensor_target = self.target_transform(target)
        else:
            tensor_target = torch.from_numpy(np.asarray(target))

        return tensor_data, tensor_target

    def __len__(self) -> int:
        return self.datas.shape[0]
