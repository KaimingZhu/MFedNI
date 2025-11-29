"""
@Encoding:      UTF-8
@File:          train_test_split.py

@Introduction:  trainset testset split for HAR dataset
@Author:        Kaiming Zhu
@Date:          2023/12/26 1:06
"""

from copy import deepcopy
import os
from typing import Tuple, List

import random
import scipy.io as sio
import numpy as np

# hyper-param: where to load original dataset
dataset_path = "./divided/"
# hyper-param: where to archive splited dataset
result_path = "./splited/"
# hyper-param: keys of each modal data
modal_keys = ["acc_feats", "gyro_feats"]
# hyper-param: testset ratio
testset_ratio = 1.0 / 6
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
