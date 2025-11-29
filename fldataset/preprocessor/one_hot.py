"""
@Encoding:      UTF-8
@File:          one_hot.py

@Introduction:  one hot preprocessor, including encoder and decoder
@Author:        Kaiming Zhu
@Date:          2023/7/30 23:06
@Reference:     https://blog.csdn.net/sinat_29957455/article/details/86552811
"""

from typing import Optional

import numpy as np

from .base_preprocessor import BasePreprocessor


class OneHotEncoder(BasePreprocessor):
    def __init__(self, max_value: Optional[int] = None):
        super().__init__()
        self.max_value = max_value

    """ Map integer Label into one-hot format. From integer format to one-hot. """
    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """ map labels to be one-hot encoded
        Examples:
            >>> labels = np.array([1, 2, 3, 2, 4])
            >>> datas = np.array([1, 2, 3, 2, 4])
            >>> encoder = OneHotEncoder()
            >>> datas, labels = encoder(datas=datas, labels=labels)
            >>> print(labels)
            [[0. 1. 0. 0. 0.]
             [0. 0. 1. 0. 0.]
             [0. 0. 0. 1. 0.]
             [0. 0. 1. 0. 0.]
             [0. 0. 0. 0. 1.]]

        See Also:
            BaseEncoder.__call__(datas:labels:)
        """
        assert (len(labels.shape) == 1, "shape of label is illegal")
        datas, labels = super().__call__(datas=datas, labels=labels)

        row_amount = self.max_value
        if row_amount is None:
            row_amount = int(np.max(labels)) + 1

        one_hot_matrix = np.eye(row_amount)
        labels = one_hot_matrix[labels]
        return datas, labels


class OneHotDecoder(BasePreprocessor):
    """ Decode one-hot encoded Label, from one-hot format to integer. """
    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """ decode labels that have been one-hot encoded.
        Examples:
            >>> labels = np.array([1, 2, 3, 2, 4])
            >>> datas = np.array([1, 2, 3, 2, 4])
            >>> encoder = OneHotEncoder()
            >>> decoder = OneHotDecoder()
            >>> datas, labels = encoder(datas=datas, labels=labels)
            >>> datas, labels = decoder(datas=datas, labels=labels)
            >>> print(labels)
            [1 2 3 2 4]

        See Also:
            BaseEncoder.__call__(datas:labels:)
        """
        assert (len(labels.shape) == 2, "shape of label is illegal")
        datas, labels = super().__call__(datas=datas, labels=labels)

        labels = np.array([int(np.argmax(label)) for label in labels])
        return datas, labels
