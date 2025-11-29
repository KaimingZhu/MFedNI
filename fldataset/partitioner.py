"""
@Encoding:      UTF-8
@File:          partitioner.py

@Introduction:  Federated Learning Data Partitioner
@Author:        Kaiming Zhu
@Date:          2023/7/26 20:16
"""

from collections.abc import Callable
from typing import List, Dict, Optional, Tuple

import numpy as np

from .dataset import Dataset
from .popular_distribution import PopularDistribution
from .distribution.sampler import BaseSampler


class Partitioner:
    def __init__(
        self,
        client_count: int,
        distribution: PopularDistribution = PopularDistribution.Dirichlet,
    ):
        """the initializer of Federated Learning Data Partitioner

        Args:
            client_count (int):                     
                The amount of clients in FL, should be ranged from [1, INT_MAX).
            distribution (PopularDistribution):
                The Distribution you want to use.

        Notes:
            Partitioner also provides a series of low-level APIs, enabling users to
            perform a customized behaviour. You can access then via following properties.

            dataset_groupped_callback(callable, optional):
                A callable object group datasets. Its type inference is:
                `Callable[[np.ndarray, np.ndarray], List[List[Tuple[np.ndarray, np.ndarray]]]]`.
                
                That means, you should accept at least two params(i.e. [datas, labels]),
                and return the groupped result in a two-dimesion list.
                - First dimension is the amount of groups.
                - Second dimension is the amount of datapoints in this group.
                - Each Datapoint is a Tuple (data, label).

            custom_sampler(BaseSampler, Optional):
                A customized sampler for deciding how to sample the dataset.
                When you set this property, `Partitioner` will ignore the
                `distribution` you have chosen. You could see `distribution.Sampler`
                for more infos.
        """

        # public: high-level API
        self.client_count: int = client_count
        self.distribution: PopularDistribution = distribution

        # public: low-level APIs, see __init__ doc strings for more infos.
        self.dataset_groupped_callback: Optional[Callable[[np.ndarray, np.ndarray], List[List[Tuple[np.ndarray, np.ndarray]]]]] = None
        self.custom_sampler: Optional[BaseSampler] = None

    def __call__(self, dataset: Dataset) -> [Dataset]:
        """divide a certain dataset to a series of subset.

        Returns:
            a list of datasets
        """
        assert (self.client_count > 0, f"client count {self.client_count} is illegal")
        datas, labels = dataset.datas, dataset.labels
        groupped_datapoints = self._group_dataset(datas=datas, labels=labels)
        clients_datapoints = self._sample_groupped_datapoints(groupped_datapoints=groupped_datapoints)
        return self._map_clients_datapoints_to_datasets(clients_datapoints=clients_datapoints)

    def _group_dataset(self, datas: np.ndarray, labels: np.ndarray) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """convert the datas and labels to the groupped datapoints.

        Args:
            datas (np.ndarray):     all characteristic in datasets.
            labels (np.ndarray):    labels of those characteristics, labels[i] means the label of datas[i].

        Returns: the groupped datapoints with type `List[List[Tuple[np.ndarray, np.ndarray]]]`
            - First dimension is the amount of groups.
            - Second dimension is the amount of datapoints in this group.
            - Each Datapoint is a Tuple (data, label).
        """
        
        if self.dataset_groupped_callback is not None:
            return self.dataset_groupped_callback(datas, labels)

        # datas with same label will be in the same group.
        datas_by_label: Dict[np.ndarray | float, List] = {}
        for i in range(0, len(datas)):
            data = datas[i]
            label = labels[i]
            datas_in_same_label = [(data, label)]

            if label not in datas_by_label.keys():
                datas_by_label[label] = datas_in_same_label
            else:
                datas_by_label[label].extend(datas_in_same_label)

        return list(datas_by_label.values())

    def _sample_groupped_datapoints(self, groupped_datapoints):
        """method to sample the groupped data"""
        if self.custom_sampler is not None:
            return self.custom_sampler(
                groupped_datapoints=groupped_datapoints,
                client_count=self.client_count
            )

        else:
            sampler = self.distribution.sampler()
            return sampler(
                groupped_datapoints=groupped_datapoints,
                client_count=self.client_count
            )

    def _map_clients_datapoints_to_datasets(
        self,
        clients_datapoints: List[List[Tuple[np.ndarray, np.ndarray]]]
    ) -> List[Dataset]:
        """map clients datapoints to clients dataset.

        Args:
            clients_datapoints(list): a two-dimension list hold by clients
                - First dimension is the data hold by each client.
                - Second dimension is the datapoints hold by corresponding client.
                - Each Datapoint is a Tuple (data, label).

        Returns:
            a list of datasets.
        """

        client_datasets: List[Dataset] = []
        for client_datapoints in clients_datapoints:
            datas = [datapoint[0] for datapoint in client_datapoints]
            labels = [datapoint[1] for datapoint in client_datapoints]

            dataset = Dataset(datas=np.array(datas), labels=np.array(labels))
            client_datasets.append(dataset)

        return client_datasets
