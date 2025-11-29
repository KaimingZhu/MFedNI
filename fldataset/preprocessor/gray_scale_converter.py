"""
@Encoding:      UTF-8
@File:          gray_scale_converter.py

@Introduction:  preprocessor for gray scale datas(e.g. Mnist, FashionMnist), enable training via `Conv2d`.
@Author:        Kaiming Zhu
@Date:          2023/8/5 23:57
"""

import numpy as np
from fldataset.preprocessor.base_preprocessor import BasePreprocessor


class GrayScaleToChannelBased(BasePreprocessor):
    """map gray scale images to channel based(i.e. `NCHW` or `NHWC`) format."""
    def __init__(self, is_channel_first: bool = True):
        self.is_channel_first = is_channel_first

    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        assert (len(datas.shape) == 3, "label's dimension should be equal to 3.")
        datas, labels = super().__call__(datas=datas, labels=labels)
        if self.is_channel_first:
            datas = datas.reshape(datas.shape[0], 1, datas.shape[1], datas.shape[2])
        else:
            datas = datas.reshape(datas.shape[0], datas.shape[1], datas.shape[2], 1)
        return datas, labels


class ChannelBasedToGrayScale(BasePreprocessor):
    """map channel based(i.e. `NCHW` or `NHWC`) images to gray scale format."""
    def __init__(self, is_channel_first: bool = True):
        self.is_channel_first = is_channel_first

    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        assert (len(datas.shape) == 4, "label's dimension should be equal to 4.")
        datas, labels = super().__call__(datas=datas, labels=labels)
        if self.is_channel_first:
            datas = np.squeeze(datas, axis=1)
        else:
            datas = np.squeeze(datas, axis=2)

        return datas, labels
