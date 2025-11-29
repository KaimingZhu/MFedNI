"""
@Encoding:      UTF-8
@File:          channel_converter.py

@Introduction:  some useful channel converter from image dataset.
@Author:        Kaiming Zhu
@Date:          2023/7/31 0:29
"""

import numpy as np

from fldataset.preprocessor.base_preprocessor import BasePreprocessor


class ChannelConverter(BasePreprocessor):
    @staticmethod
    def is_channel_first(datas: np.ndarray) -> bool:
        """Judge if any image datas is formatted with `NCHW`, commonly used for `RGB` images."""
        if len(datas.shape) != 4:
            return False
        return datas.shape[1] == 3

    @staticmethod
    def is_channel_last(datas: np.ndarray) -> bool:
        """Judge if any image datas is formatted with `NHWC`, commonly used for `RGB` images."""
        if len(datas.shape) != 4:
            return False
        return datas.shape[-1] == 3


class ChannelLastToFirst(ChannelConverter):
    """map image dataset from `NHWC` format to `NCHW` format."""
    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        See Also:
            BasePreprocessor.__call__(datas:labels:)
        """
        assert (len(datas.shape) == 4, "label's dimension must equal to 4.")
        datas, labels = super().__call__(datas=datas, labels=labels)
        datas = np.transpose(datas, (0, 3, 1, 2))

        return datas, labels


class ChannelFirstToLast(ChannelConverter):
    """map image dataset from `NCHW` format to `NHWC` format."""
    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        See Also:
            BasePreprocessor.__call__(datas:labels:)
        """
        assert (len(datas.shape) == 4, "label's dimension must equal to 4.")
        datas, labels = super().__call__(datas=datas, labels=labels)
        datas = np.transpose(datas, (0, 2, 3, 1))

        return datas, labels