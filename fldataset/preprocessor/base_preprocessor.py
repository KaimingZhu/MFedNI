"""
@Encoding:      UTF-8
@File:          base_preprocessor.py

@Introduction:  the definition of BasePreprocessor
@Author:        Kaiming Zhu
@Date:          2023/7/30 22:50
"""

import numpy as np


class BasePreprocessor:
    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """make a copy of datas and labels, should be overriden by subclasses to make further mapping.

        BasePreprocessor will make identity-conversion, i.e. just make a copy
        for these two arrays and return. For all subclasses, please call
        `super().__call__(datas=datas, labels=labels)` first, to ensure the
        arrays you map is different from before.

        Args:
            datas(np.ndarray): Datas in dataset. First dimension should be the index of each data.
            labels(np.ndarray): Labels in dataset. First dimension should be the label of datas[i].

        Returns:
            A Tuple that indicates mapped (datas, labels).
        """

        new_datas = datas.copy()
        new_labels = labels.copy()
        return new_datas, new_labels
