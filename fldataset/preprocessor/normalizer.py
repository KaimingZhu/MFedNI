"""
@Encoding:      UTF-8
@File:          normalizer.py

@Introduction:  definition of some general normalization on datas
@Author:        Kaiming Zhu
@Date:          2023/7/31 0:11
@Reference:     https://zhuanlan.zhihu.com/p/424518359
"""
from typing import Optional

import numpy as np

from fldataset.preprocessor.base_preprocessor import BasePreprocessor


class MinMaxNormalizer(BasePreprocessor):
    """ normalize datas within the min-max value.
        data_{i} = \frac{data_{i} - min_value}{max_value - min_value}.
    """
    def __init__(self, min: Optional[float] = None, max: Optional[float] = None):
        self.min_value = min
        self.max_value = max

    def __call__(self, datas: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """a normalizer to map datas w.r.t. min value and max value.
        Notes:
            for `min` and `max`, if one of them has given, normalizer
            will take it as the min / max value for calculation, otherwise
            get it from np.max() or np.min()

        See Also:
            BasePreprocessor.__call__(datas:labels:)
        """

        datas, labels = super().__call__(datas=datas, labels=labels)

        min_value: float = self.min_value
        if min_value is None:
            min_value = np.min(datas)
        max_value: float = self.max_value
        if max_value is None:
            max_value = np.max(datas)

        datas = datas.astype('float32')
        datas = (datas - min_value) / (max_value - min_value)
        return datas, labels
