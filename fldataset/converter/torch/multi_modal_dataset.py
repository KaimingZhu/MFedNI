"""
@Encoding:      UTF-8
@File:          multi_modal_dataset.py

@Introduction:  Definition of `MultiModalDataset`
@Author:        Kaiming Zhu
@Date:          2023/12/21 0:54
@Reference:     https://discuss.pytorch.org/t/how-to-use-dataset-class-to-load-a-multimodal-dataset/172177
"""

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    data_keys = []
    target_key = "label"

    def __init__(self, modal_data_by_key: Dict[str, np.ndarray], targets: np.ndarray):
        self.modal_data_by_key = modal_data_by_key
        self.targets = targets
        self.data_keys = sorted(modal_data_by_key.keys())

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for key, modal_data in self.modal_data_by_key.items():
            result[key] = torch.from_numpy(modal_data[index]).float()
        result[self.target_key] = torch.from_numpy(np.asarray(self.targets[index]).astype('int64'))
        return result

    def __len__(self) -> int:
        return self.targets.shape[0]
