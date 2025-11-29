"""
@Encoding:      UTF-8
@File:          make_modal_incomplete_dataset.py

@Introduction:  make modal incomplete dataset w.r.t. original one
@Author:        Kaiming Zhu
@Date:          2023/12/6 22:13
@Reference:     https://blog.csdn.net/Roxlu7/article/details/126444256
"""

from copy import deepcopy
import os
import random
from typing import List, Callable, Optional
from collections import Counter

import scipy.io as sio
import numpy as np

# hyper-param: where to load original dataset
dataset_path = "./origin/"
# hyper-param: where to archive incomplete dataset
result_path = "./incomplete/"
# hyper-param: keys of each modal data
modal_keys = ["acc_feats", "gyro_feats"]
# hyper-param: probability of data will be modal-complete, set it as 'None' if you need uniform random.
complete_probability: Optional[float] = 0.14
# hyper-param: specified random seed, set it as 'None' if you do not need it
seed: Optional[float] = None

# counter for code has been generated
generated_numbers = []


def make_modal_incompleteness(modal_datas: List, random_num_generator: Callable):
    if len(modal_datas) == 0:
        return []

    origin_shapes = [data.shape for data in modal_datas]
    modal_datas = [data.reshape(-1, data.shape[-1]) for data in modal_datas]

    data_amount = modal_datas[0].shape[0]
    for index in range(0, data_amount):
        random_num = random_num_generator(lowerbound=1, upbound=2**len(modal_datas)-1)

        # This Line is just for counting, debug only.
        generated_numbers.append(random_num)

        for modal in modal_datas:
            should_abort = (random_num % 2) == 0
            random_num = random_num >> 1
            if not should_abort:
                continue
            modal[index].fill(0)

    return [data.reshape(shape) for data, shape in zip(modal_datas, origin_shapes)]


def uniform_random(lowerbound: int, upbound: int) -> int:
    return random.randint(lowerbound, upbound)


def complete_bias_random(lowerbound: int, upbound: int) -> int:
    if complete_probability >= random.random():
        return upbound
    else:
        return random.randint(lowerbound, upbound-1)


if __name__ == "__main__":
    # fixed random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if complete_probability is not None:
        method = complete_bias_random
    else:
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
    # The existence of modal acc_feats is encoded as: 1
    # The existence of modal gyro_feats is encoded as: 10
    #
    # Results:
    # 10 has generated 1349 times
    # 11 has generated 2566 times
    # 1 has generated 1324 times
