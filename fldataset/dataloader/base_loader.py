"""
@Encoding:      UTF-8
@File:          base_loader.py

@Introduction:  An abstract class describing the `BaseLoader`
@Author:        Kaiming Zhu
@Date:          2023/7/19 21:00
"""

from typing import Tuple
from abc import ABCMeta, abstractclassmethod, abstractmethod
import sys

import numpy as np

from fldataset.public_dataset import PublicDataset


class BaseLoader(metaclass=ABCMeta):
    @abstractclassmethod
    def is_available(cls) -> bool:
        """Return `true` to indicate that this loader is available."""
        pass

    @abstractmethod
    def load_public_dataset(
        self,
        dataset: PublicDataset
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """load public dataset by enum defined in `Dataset`
        Returns:
            dataset tuple, formatted with (train_datas, train_labels), (test_datas, test_labels).
        """
        pass

    @classmethod
    def _is_module_imported(cls, name: str) -> bool:
        system_keys = list(sys.modules.keys())
        return name in system_keys

    @classmethod
    def _dataset_cache_path(cls) -> str:
        """A str indicating the root path to cache data.

        When you fetch popular datasets (e.g. Mnist, Cifar10) by APIs provided by popular
        machine earning Framework(e.g. TensorFlow, Pytorch), they will cache those datasets
        to the path you specified.

        When you call the same api with same cache path, they will check out whether the
        dataset has cached in the path you gave. If so, it will just load it without
        re-fetching, speeding up your program.

        This function is the restriction of root caching path.
        For each loader, the path they use will be `"./.dataset/{Loader.name()}/"
        """
        return "./.dataset/" + cls.name() + "/"

    @classmethod
    def name(cls) -> str:
        """
        Name of loader, will be used for data caching.
        all subclasses should override it.
        """
        return "base"
