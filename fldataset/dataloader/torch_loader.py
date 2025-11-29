"""
@Encoding:      UTF-8
@File:          torch_loader.py

@Introduction:  loader for pytorch
@Author:        Kaiming Zhu
@Date:          2023/7/19 21:10
@Reference:     https://stackoverflow.com/questions/54897646/pytorch-datasets-converting-entire-dataset-to-numpy
"""
from typing import Tuple, Any

import numpy as np


from .base_loader import BaseLoader
from ..public_dataset import PublicDataset


class TorchLoader(BaseLoader):
    @classmethod
    def name(cls) -> str:
        return "torch"

    @classmethod
    def is_available(cls) -> bool:
        """
        See also:
            BaseLoader.is_available
        """
        return cls._is_module_imported(name="torch") or cls._is_module_imported(name="torchvision")

    def load_public_dataset(
        self,
        dataset: PublicDataset
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        See also:
            BaseLoader.load_public_dataset
        """
        from torchvision import datasets as datasets
        from torchvision.datasets.vision import VisionDataset
        from torchvision.transforms import ToTensor

        dataset_cls_by_enum: {PublicDataset: VisionDataset} = {
            PublicDataset.Mnist: datasets.MNIST,
            PublicDataset.FashionMnist: datasets.FashionMNIST,
            PublicDataset.Cifar10: datasets.CIFAR10,
            PublicDataset.Cifar100: datasets.CIFAR100
        }

        dataset_cls = dataset_cls_by_enum[dataset]
        assert (dataset_cls is not None, f"fail to load dataset with {dataset.value}")

        cache_path = self.__class__._dataset_cache_path() + f"{dataset.value}/"
        trainset = dataset_cls(
            root=cache_path + "train/",
            train=True,
            download=True,
            transform=ToTensor()
        )
        testset = dataset_cls(
            root=cache_path + "test/",
            train=False,
            download=True,
            transform=ToTensor()
        )

        (train_datas, train_labels) = self.__unwrap_dataset_from_vision_dataset(dataset=trainset)
        (test_datas, test_labels) = self.__unwrap_dataset_from_vision_dataset(dataset=testset)

        train_datas = self.__post_process_for_datas(datas=train_datas, dataset=dataset)
        test_datas = self.__post_process_for_datas(datas=test_datas, dataset=dataset)
        train_labels = self.__post_process_for_labels(labels=train_labels, dataset=dataset)
        test_labels = self.__post_process_for_labels(labels=test_labels, dataset=dataset)

        return (train_datas, train_labels), (test_datas, test_labels)

    def __unwrap_dataset_from_vision_dataset(self, dataset) -> ([np.ndarray], [np.ndarray]):
        "a wrapped function map `VisionDataset` to `(Numpy_datas, Numpy_labels)."
        from torchvision.datasets import VisionDataset

        assert (isinstance(dataset, VisionDataset),
                "Expect type VisionDataset, but receive "
                f"an obj with type {type(dataset).__name__}")

        datas = self.__unwrap_torch_return_val_to_numpy(dataset.data)
        labels = self.__unwrap_torch_return_val_to_numpy(dataset.targets)

        return (datas, labels)

    def __unwrap_torch_return_val_to_numpy(self, return_val: Any) -> np.ndarray:
        if isinstance(return_val, np.ndarray):
            return return_val

        if isinstance(return_val, list):
            return np.array(return_val)

        from torch import Tensor
        if isinstance(return_val, Tensor):
            return return_val.numpy()

        raise RuntimeError(f"fail to convert an return val with dtype {type(return_val)} into np.ndarray")

    def __post_process_for_datas(self, datas: np.ndarray, dataset: PublicDataset):
        return datas

    def __post_process_for_labels(self, labels: np.ndarray, dataset: PublicDataset):
        # A post process adpating for Cifar10 and Cifar100, to fix the bug raised in nn.CrossEntropyLoss:
        # "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'.
        if dataset == PublicDataset.Cifar10 or dataset == PublicDataset.Cifar100:
            labels = labels.astype('int64')
        return labels

