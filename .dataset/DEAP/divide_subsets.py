"""
@Encoding:      UTF-8
@File:          divide_subsets.py

@Introduction:  script to divide subsets for DEAP dataset
@Author:        Kaiming Zhu
@Date:          2024/1/6 4:24
"""

import os
import random
import sys

import scipy.io as sio
import numpy as np

sys.path.append("../..")
from fldataset.distribution import NonIIDSampler, DirichletDistributed, IIDSampler, UniformDistributed


# hyper-param: where to load original dataset
dataset_path = "./window_slided/"
# hyper-param: where to archive divided dataset
result_path = "./divided/"
# hyper-param: keys of each modal
modal_keys = ["eeg_feats", "emg_feats", "eog_feats"]
# hyper-param: keys of labels
label_key = "labels"
# hyper-param: client counts
client_counts = 20
# hyper-param: alpha for dirichlet distribution(Non-IID), set it as `None` if you need IID sampling
alpha = None
# hyper-param: specified random seed, set it as 'None' if you do not need it
seed = None


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
    #    - First dimension is the group of datapoints, where all elems in group have same label.
    #    - Second dimension is datapoints in this group.
    #    - Datapoint will be a tuple like (modal1, modal2, ..., label).
    groupped_datapoints = [[] for _ in range(max(labels) + 1)]

    for index, label in enumerate(labels):
        datas_at_index = [modal_datas[index] for modal_datas in datas]
        datapoint = datas_at_index + [label]
        groupped_datapoints[label].append(tuple(datapoint))

    # divide dataset into subsets
    each_client_datapoints = []
    if alpha is not None:
        distribution = DirichletDistributed(alpha=alpha)
        sampler = NonIIDSampler(distribution_generator=distribution)
    else:
        distribution = UniformDistributed()
        sampler = IIDSampler(distribution_generator=distribution)
    each_client_datapoints = sampler(groupped_datapoints=groupped_datapoints, client_count=client_counts)

    # reshape result, and save to subset
    for client_index, client_datapoints in enumerate(each_client_datapoints):
        client_datas: [np.ndarray] = [[] for _ in modal_keys]
        client_labels: np.ndarray = []

        for datapoint in client_datapoints:
            for elem, client_modal_datas in zip(datapoint[0:-1], client_datas):
                client_modal_datas.append(elem)
            label = datapoint[-1]
            client_labels.append(label)

        client_datas = [np.stack(modal_datas) for modal_datas in client_datas]
        client_labels = np.array(client_labels)
        for key, modal_datas in zip(modal_keys, client_datas):
            dataset_dict[key] = modal_datas
        dataset_dict[label_key] = client_labels

        sio.savemat(result_path + f"client{client_index}.mat", dataset_dict)
