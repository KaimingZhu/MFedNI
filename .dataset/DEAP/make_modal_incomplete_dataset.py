"""
@Encoding:      UTF-8
@File:          make_modal_incomplete_dataset.py

@Introduction:  make modal incomplete dataset w.r.t. original one, for DEAP dataset
@Author:        Kaiming Zhu
@Date:          2024/1/6 11:27
"""

from copy import deepcopy
import os
import random
from typing import List, Callable
from collections import Counter

import scipy.io as sio
import numpy as np

# hyper-param: where to load original dataset
dataset_path = "./origin/"
# hyper-param: where to archive incomplete dataset
result_path = "./incomplete/"
# hyper-param: keys of each modal data
modal_keys = ["eeg_feats", "emg_feats", "eog_feats"]
# hyper-param: specified random seed, set it as 'None' if you do not need it
seed = None


# fixed random seed
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)

# counter for code has been generated
generated_numbers = []


def make_modal_incompleteness(modal_datas: List, random_num_generator: Callable):
    if len(modal_datas) == 0:
        return []

    experiment_amount = modal_datas[0].shape[0]
    timestamp_amount = modal_datas[0].shape[1]
    for experiment in range(0, experiment_amount):
        for timestamp in range(0, timestamp_amount):
            random_num = random_num_generator(lowerbound=1, upbound=2 ** len(modal_datas) - 1)
            # This Line is just for counting, debug only.
            generated_numbers.append(random_num)

            for modal in modal_datas:
                should_abort = (random_num % 2) == 0
                random_num = random_num >> 1
                if not should_abort:
                    continue
                modal[experiment][timestamp].fill(0)

    return modal_datas


def uniform_random(lowerbound: int, upbound: int) -> int:
    return random.randint(lowerbound, upbound)


def complete_bias_random(lowerbound: int, upbound: int, bias_probability: float = 0.5) -> int:
    if bias_probability >= random.random():
        return upbound
    else:
        return random.randint(lowerbound, upbound-1)


if __name__ == "__main__":
    method = uniform_random

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    filenames = list(os.listdir(dataset_path))
    filenames = [filename for filename in filenames if filename.endswith(".mat")]
    filenames.sort()
    for filename in filenames:
        data_by_name = sio.loadmat(dataset_path + filename, squeeze_me=True)
        data_by_name = deepcopy(data_by_name)
        modal_datas = [data_by_name[key] for key in modal_keys]

        modal_datas = make_modal_incompleteness(
            modal_datas=modal_datas,
            random_num_generator=method
        )
        for index, key in enumerate(modal_keys):
            data_by_name[key] = modal_datas[index]

        sio.savemat(result_path + filename, data_by_name)

    print("Encodes:")
    for index, name in enumerate(modal_keys):
        print(f"The existence of modal {name} is encoded as: " + bin(2 ** index)[2:])

    print("\nResults:")
    counter = Counter(generated_numbers)
    for key, value in counter.items():
        print(bin(key)[2:] + f" has generated {value} times")

    # ** Output Example **
    # Encodes:
    # The existence of modal eeg_feats is encoded as: 1
    # The existence of modal emg_feats is encoded as: 10
    # The existence of modal eog_feats is encoded as: 100
    #
    # Results:
    # 110 has generated 5377 times
    # 111 has generated 5462 times
    # 11 has generated 5498 times
    # 101 has generated 5490 times
    # 100 has generated 5622 times
    # 10 has generated 5457 times
    # 1 has generated 5494 times

