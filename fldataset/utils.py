"""
@Encoding:      UTF-8
@File:          utils.py

@Introduction:  useful utils for fldataset
@Author:        Kaiming Zhu
@Date:          2023/12/22 2:09
"""

import random
from typing import Tuple, List


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