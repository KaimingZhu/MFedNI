"""
@Encoding:      UTF-8
@File:          divide_subsets.py

@Introduction:  script to divide subsets for HAR dataset
@Author:        Kaiming Zhu
@Date:          2023/12/28 23:47
"""

import os
import random
import sys
from typing import Optional

import scipy.io as sio
import numpy as np

sys.path.append("../..")
from fldataset.distribution import NonIIDSampler, DirichletDistributed, IIDSampler, UniformDistributed


# hyper-param: where to load original dataset
dataset_path = "./window_slided/"
# hyper-param: where to archive divided dataset
result_path = "./divided/"
# hyper-param: keys of each modal
modal_keys = ["acc_feats", "gyro_feats"]
# hyper-param: keys of labels
label_key = "labels"
# hyper-param: client counts
client_counts: int = 20
# hyper-param: alpha for dirichlet distribution(Non-IID), set it as `None` if you need IID sampling.
alpha: Optional[float] = 1.0
# hyper-param: specified random seed, set it as 'None' if you do not need it
seed: Optional[int] = None
# hyper-param: a boolean flag to indicate whether users can hold an empty dataset after division.
is_user_dataset_empty_allowed = False


if seed is not None:
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 1. load original dataset
    filenames = list(os.listdir(dataset_path))
    filenames = [filename for filename in filenames if filename.endswith(".mat")]
    dataset_dict = sio.loadmat(dataset_path + filenames[0], squeeze_me=True)
    datas: [np.ndarray] = [dataset_dict[key] for key in modal_keys]
    labels: np.ndarray = dataset_dict[label_key]

    # 2. group dataset into a two-dimension list
    #      - First dimension is the group of datapoints, where all elems in group have same label.
    #      - Second dimension is datapoints in this group.
    #      - Datapoint will be a tuple like (modal1, modal2, ..., label).
    groupped_datapoints = [[] for _ in range(max(labels) + 1)]
    while len(labels) > 0:
        # make iteration on Labels, find first index which is not continuous, and store at first_not_continuous_index.
        current_label = labels[0]
        not_continuous_indices = np.where(labels != current_label)[0]
        if len(not_continuous_indices) == 0:
            first_not_continuous_index = len(labels)
        else:
            first_not_continuous_index = not_continuous_indices[0]

        # get slice of datapoint
        datapoint = []
        for modal_datas in datas:
            datapoint.append(modal_datas[0:first_not_continuous_index])
        datapoint.append(current_label)

        # added into corresponding group
        groupped_datapoints[current_label].append(tuple(datapoint))

        # truncate datas and labels
        datas = [modal_datas[first_not_continuous_index:] for modal_datas in datas]
        labels = labels[first_not_continuous_index:]

    # divide dataset into subsets
    each_client_datapoints = []
    while len(each_client_datapoints) == 0:
        if alpha is not None:
            distribution = DirichletDistributed(alpha=alpha)
            sampler = NonIIDSampler(distribution_generator=distribution)
        else:
            distribution = UniformDistributed()
            sampler = IIDSampler(distribution_generator=distribution)
        each_client_datapoints = sampler(groupped_datapoints=groupped_datapoints, client_count=client_counts)

        if not is_user_dataset_empty_allowed:
            sizes = [len(datapoints) for datapoints in each_client_datapoints]
            is_each_equal_to_zero = [size == 0 for size in sizes]
            if any(is_each_equal_to_zero):
                each_client_datapoints = []

    # reshape result, and save to subset
    for client_index, client_datapoints in enumerate(each_client_datapoints):
        client_datas: [np.ndarray] = [[] for _ in modal_keys]
        client_labels: np.ndarray = []

        for datapoint in client_datapoints:
            for elem, client_modal_datas in zip(datapoint[0:-1], client_datas):
                client_modal_datas.append(elem)
            data_amount = len(client_datas[0][-1])
            label = datapoint[-1]
            client_labels.append(np.array([label for _ in range(data_amount)]))

        client_datas = [np.concatenate(modal_datas) for modal_datas in client_datas]
        client_labels = np.concatenate(client_labels)
        for key, modal_datas in zip(modal_keys, client_datas):
            dataset_dict[key] = modal_datas
        dataset_dict[label_key] = client_labels

        sio.savemat(result_path + f"client{client_index}.mat", dataset_dict)