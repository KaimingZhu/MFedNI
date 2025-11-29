"""
@Encoding:      UTF-8
@Filename:      sampler.py

@Introduction:  The Proportion Generator
@Author:        Kaiming Zhu
@Date:          2023/07/12 19:27
"""

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np

from .distribution_generator import BaseDistributionGenerator, DirichletDistributed, UniformDistributed


class BaseSampler(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self,
        groupped_datapoints: List[List[Tuple[np.ndarray, np.ndarray]]],
        client_count: int
    ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """Divide the groupped datapoints with respect to client_num.

        Args:
            groupped_datapoints(list):
                - First dimension is the amount of groups.
                - Second dimension is the amount of datapoints in this group.
                - Each Datapoint is a Tuple (data, label).
            client_count(int):
                an integer indicating the amount of clients, ranged from [0, INT_MAX)

        Returns: a two-dimension list, indicating the result of sampling.
            - First dimension is the data hold by each client.
            - Second dimension is the datapoints hold by corresponding client.
            - Each Datapoint is a Tuple (data, label).
        """
        pass


class IIDSampler(BaseSampler):
    """IID Sampling"""
    def __init__(self, distribution_generator: BaseDistributionGenerator = UniformDistributed()):
        self.distribution_generator = distribution_generator

    def __call__(
        self,
        groupped_datapoints: List[List[Tuple[np.ndarray, np.ndarray]]],
        client_count: int
    ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        See Also:
            BaseSampler.__call__
        """
        result = [[] for _ in range(0, client_count)]
        distribution = self.distribution_generator(client_num=client_count)

        for data_label_tuples in groupped_datapoints:
            each_client_datas = self.sample(datas=data_label_tuples, distribution=distribution)
            for client in range(0, client_count):
                result[client].extend(each_client_datas[client])

        return result

    def sample(self, datas: list, distribution: np.ndarray) -> list:
        """Sample datas with respect to(w.r.t.) a certain distribution."""

        # Calculate all clients sample amount w.r.t. distribution.
        # As `np.round` will lead to a round-off error,
        # we will let the last client to hold these extra data.
        each_client_sample_amount = np.round(len(datas) * distribution)
        each_client_sample_amount[-1] = len(datas) - np.sum(each_client_sample_amount[0:-1])

        each_client_start_index = [0]
        for sample_amount in each_client_sample_amount:
            each_client_start_index.append(each_client_start_index[-1] + sample_amount)

        client_amount = len(distribution)
        each_client_datas = []
        for client in range(0, client_amount):
            sampling_start_index = int(each_client_start_index[client])
            sampling_end_index = int(each_client_start_index[client + 1])
            sampling_slice = slice(sampling_start_index, sampling_end_index)

            client_data = datas[sampling_slice]
            each_client_datas.append(client_data)

        return each_client_datas


class NonIIDSampler(IIDSampler):
    """NonIID Sampling"""
    def __init__(self, distribution_generator: BaseDistributionGenerator = DirichletDistributed()):
        super().__init__(distribution_generator=distribution_generator)

    def __call__(
        self,
        groupped_datapoints: List[List[Tuple[np.ndarray, np.ndarray]]],
        client_count: int
    ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        See Also:
            BaseSampler.__call__
        """
        result = [[] for _ in range(0, client_count)]

        for data_group in groupped_datapoints:
            distribution = self.distribution_generator(client_num=client_count)
            each_client_datas = self.sample(datas=data_group, distribution=distribution)
            for client in range(0, client_count):
                result[client].extend(each_client_datas[client])

        return result


class ShardingSampler(BaseSampler):
    """
    Sharding Sampling
    """

    def __call__(
        self,
        groupped_datapoints: List[List[Tuple[np.ndarray, np.ndarray]]],
        client_count: int
    ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        raise NotImplementedError("Would be supported in future.")