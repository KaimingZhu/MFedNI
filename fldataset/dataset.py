"""
@Encoding:      UTF-8
@File:          dataset.py

@Introduction:  The definition of fldataset.Dataset
@Author:        Kaiming Zhu
@Date:          2023/7/29 7:18
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple, Callable
import os

import numpy as np

from .public_dataset import PublicDataset
from .dataloader import *
from .preprocessor.base_preprocessor import BasePreprocessor


class Dataset:
    __loader_classes = [TorchLoader]

    def __init__(self, datas: np.ndarray, labels: np.ndarray):
        """initializer of Dataset.
        Args:
            datas(np.ndarray):
                Datas in dataset. First dimension should be the
                index of each data.
            labels(np.ndarray):
                labels in dataset. First dimension should be the
                label of datas[i].

        Warnings:
             If datas.shape[0] != labels.shape[0], it will throws
             an error.
        """

        self._assert_is_instance(expect_type=np.ndarray, obj=datas)
        self._assert_is_instance(expect_type=np.ndarray, obj=labels)
        assert (datas.shape[0] == labels.shape[0], "first dimension of data and label is not equal.")

        self.datas = datas
        self.labels = labels

    @classmethod
    def load_public_dataset(cls, dataset: PublicDataset) -> (Dataset, Dataset):
        """load public dataset by their name.
        Notes:
            Before you call this method, please import essential
            libs(e.g. TensorFlow, Keras, Pytorch) first.

        Args:
            name(Dataset): the dataset you want to load.

        Returns:
            A Dataset tuple (trainset, testset).

        Warnings:
            This method will throw an exception if any condition meets:
                - fail to find any suitable libs to fetch it.
                - all these suitable loaders fail to load the data you need.
        """
        available_loader_clses = list(filter(
            lambda loader_cls: loader_cls.is_available(),
            cls.__loader_classes)
        )

        if len(available_loader_clses) == 0:
            assert (False, "Fail to find suitable env for any loader, "
                           "please import essential libs"
                           "(e.g. tensorflow, torch) first.")

        trainset_testset_tuple: Optional[Tuple] = None
        available_loaders = [cls() for cls in available_loader_clses]
        for loader in available_loaders:
            trainset_testset_tuple = loader.load_public_dataset(dataset=dataset)
            if trainset_testset_tuple is not None:
                break
        else:
            assert (False, f"All loaders fail to load {dataset.value}")

        trainset_tuple = trainset_testset_tuple[0]
        testset_tuple = trainset_testset_tuple[1]
        trainset = Dataset(datas=trainset_tuple[0], labels=trainset_tuple[1])
        testset = Dataset(datas=testset_tuple[0], labels=testset_tuple[1])

        return trainset, testset

    def __deepcopy__(self, memodict: Optional[Dict] = None) -> Dataset:
        if memodict is None:
            memodict = {}

        datas = self.datas.copy()
        labels = self.labels.copy()
        return Dataset(datas=datas, labels=labels)

    def __add__(self, other: Optional[Dataset]) -> Dataset:
        """magic method to merge two dataset.
        See Also:
            Dataset.__add__(self, other)
        """
        if other is None:
            return copy.deepcopy(self)

        new_dataset = copy.deepcopy(self)
        new_dataset += other
        return new_dataset

    def __iadd__(self, other: Optional[Dataset]) -> Dataset:
        """magic method to merge another dataset.
        Args:
            other(Dataset):
                a dataset which **MUST** obey following conditions
                (1) self.datas.shape[1:] == other.datas.shape[1:]
                (2) self.labels.shape[1:] == other.labels.shape[1:]

        Returns:
            A new Dataset object which has merged two of them,
            datas and labels in `other` will be appended to the
            bottom of original.

        Warnings:
            If rules declare in `Args` is broken, this method will
            throw an error.
        """
        if other is None:
            return self

        self.datas = np.concatenate((self.datas, other.datas), axis=0)
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        return self

    def __len__(self):
        """magic method to calculate dataset length."""
        return len(self.datas)

    def shuffle(self):
        """ Method to shuffle current dataset. """
        index = np.arange(0, len(self.datas))
        np.random.shuffle(index)
        self.datas = self.datas[index]
        self.labels = self.labels[index]

    def shuffled(self) -> Dataset:
        """Method to get a new dataset, which is shuffled base on current dataset."""
        dataset = Dataset(datas=self.datas.copy(), labels=self.labels.copy())
        dataset.shuffle()
        return dataset

    def split(self, ratio: float) -> (Dataset, Dataset):
        """split dataset with respect to the ratio you have given.
        Args:
            ratio(float): range from [0,1].

        Returns:
            a tuple (firstset, secondset). The scale of firstset equals to
            `ratio`, while the secondset equals to `(1 - ratio)`.
        """

        slice_index = int(np.round(len(self.datas) * ratio))
        first_set = Dataset(
            datas=self.datas[0:slice_index].copy(),
            labels=self.labels[0:slice_index].copy()
        )
        second_set = Dataset(
            datas=self.datas[slice_index:].copy(),
            labels=self.labels[slice_index:].copy()
        )
        return first_set, second_set

    @classmethod
    def load_from(cls, folder: str, filename: str = "dataset") -> Dataset:
        """load Dataset with respect to(w.r.t.) the path you gave.
        Examples:
            # For dataset has archived at "./dataset.npz"
            Dataset.load_from(folder=".", filename="dataset")
            # For dataset has archived at "./test/extra_set.npz"
            Dataset.load_from(folder="./test", filename="extra_set")

        Args:
            folder(str): the folder that archived the dataset file.
            filename(str): name of the archived dataset, will be load as f"{filename.npz}".

        Returns:
            The corresponding Dataset object, load from the path you gave.

        Warnings:
            method will raise error if path illegal, or fail to read.

        See Also:
            Dataset.save_to(self, folder, filename)
        """
        if folder[-1] == "/" or folder[-1] == "\\":
            folder = folder[:-1]
        assert (os.path.isdir(folder), f"{folder} is illegal")
        assert (os.path.exists(folder), f"{folder} do not exist")

        path = f"{folder}/{filename}.npz"
        datas: Optional[np.ndarray] = None
        labels: Optional[np.ndarray] = None
        with np.load(path, allow_pickle=True) as f:
            datas = f["datas"]
            labels = f["labels"]

        assert (datas is not None, f"fail to load datas in {path}")
        assert (labels is not None, f"fail to load labels in {path}")
        return cls(datas=datas, labels=labels)

    def save_to(self, folder: str, filename: str = "dataset"):
        """save dataset to the path you gave.
        When you call this method Dataset will be saved in f"{folder}/{filename}.npz".
        The key-value in this npz will be:
            "datas": a `np.ndarray` object, indicating the datas in dataset.
            "labels": a `np.ndarray` object, indicating the labels in dataset.

        Args:
            folder(str): folder you want to archive the dataset.
            filename(str): name of the archive file, will be saved as f"{filename}.npz".

        Warnings:
            this method will raise an error if path illegal.
        """
        assert (os.path.isdir(folder), f"{folder} is illegal")
        if not os.path.exists(folder):
            os.makedirs(folder)

        np.savez(file=f"{folder}/{filename}.npz", datas=self.datas, labels=self.labels)

    def map(self, preprocessor: BasePreprocessor) -> Dataset:
        """preprocess the dataset with any preprocessor, and return the result.
        Args:
            preprocessor(BasePreprocessor): The Preprocessor you need.

        Returns:
            a new dataset has been preprocessed with it.
        """

        datas, labels = preprocessor(datas=self.datas, labels=self.labels)
        return Dataset(datas=datas, labels=labels)

    def _assert_is_instance(self, expect_type: Any, obj: Any):
        assert (
            isinstance(obj, expect_type),
            f"expect {expect_type.__name__}, but receive an object with"
            f"type: {type(obj).__class__.__name__}"
        )
