"""
@Encoding:      UTF-8
@File:          train_test_split.py

@Introduction:  trainset testset split for DEAP dataset
@Author:        Kaiming Zhu
@Date:          2023/12/26 2:01
"""

from copy import deepcopy
import os
from typing import Tuple, List

import random
import scipy.io as sio
import numpy as np


# hyper-param: where to load original dataset
dataset_path = "./divided/"
# hyper-param: where to archive splitted dataset
result_path = "./splitted/"
# hyper-param: keys of each modal data
modal_keys = ["eeg_feats", "emg_feats", "eog_feats"]
# hyper-param: testset ratio
testset_ratio = 1.0 / 6
# hyper-param: boolean flag to indicate if you need flatten it.
# hints: plz flatten for non-window slided dataset.
need_flatten = False
# hyper-param: specified random seed, set it as 'None' if you do not need it
seed = None


# fixed random seed
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)


def shuffle_lists_with_same_indices(*lists: Tuple[List]):
    if len(lists) == 0:
        return lists

    new_lists = [[] for _ in lists]
    new_indices = list(range(0, len(lists[0])))
    random.shuffle(new_indices)
    for new_index in new_indices:
        for sublist_index, new_sublist in enumerate(new_lists):
            new_sublist.append(lists[sublist_index][new_index])

    return tuple(new_lists)


def flatten_dataset(datas: [np.ndarray], labels: np.ndarray):
    shapes = [modal_datas.shape for modal_datas in datas]
    shape_length = [len(shape) for shape in shapes]
    need_flattens = [length > 2 for length in shape_length]
    if not all(need_flattens):
        return datas, labels

    result_datas = [[] for _ in datas]
    result_labels = []
    for index, label in enumerate(labels):
        data_at_index = [modal_datas[index] for modal_datas in datas]
        flatten_data_at_index = [modal_data.reshape(-1, modal_data.shape[-1]) for modal_data in data_at_index]
        for result_modal_datas, flatten_modal_data in zip(result_datas, flatten_data_at_index):
            result_modal_datas.append(flatten_modal_data)

        label_amount = flatten_data_at_index[0].shape[0]
        result_labels.extend([label for _ in range(label_amount)])

    result_datas = [np.concatenate(modal_datas) for modal_datas in result_datas]
    result_labels = np.array(result_labels)
    return result_datas, result_labels



if __name__ == "__main__":
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    filenames = list(os.listdir(dataset_path))
    filenames = [filename for filename in filenames if filename.endswith(".mat")]
    filenames.sort()
    for filename in filenames:
        # Step 1. Load Dataset, store at 'modal_datas' and 'labels'
        data_by_name = sio.loadmat(dataset_path + filename, squeeze_me=True)
        data_by_name = deepcopy(data_by_name)
        datas: [np.ndarray] = [data_by_name[key] for key in modal_keys]
        labels: np.ndarray = data_by_name["labels"]

        if need_flatten:
            datas, labels = flatten_dataset(datas=datas, labels=labels)

        # Step 2. shuffle
        *datas, labels = shuffle_lists_with_same_indices(*(datas + [labels]))
        datas = [np.array(data) for data in datas]
        labels = np.array(labels)

        # Step 3. split w.r.t. testset_ratio
        split_index = round(len(labels) * (1.0 - testset_ratio))
        train_datas = [data[0:split_index] for data in datas]
        train_labels = labels[0:split_index]
        test_datas = [data[split_index:] for data in datas]
        test_labels = labels[split_index:]

        # Step 4. store them to `data_by_name`
        #    - trainset -> modal_keys
        #    - testset -> "test_" + modal_keys
        for key, modal_datas in zip(modal_keys, train_datas):
            data_by_name[key] = modal_datas
        data_by_name["labels"] = train_labels

        testset_keys = ["test_" + key for key in modal_keys]
        for key, modal_datas in zip(testset_keys, test_datas):
            data_by_name[key] = modal_datas
        data_by_name["test_labels"] = test_labels

        sio.savemat(result_path + filename, data_by_name)
