"""
@Encoding:      UTF-8
@File:          path.py

@Introduction:  path utils
@Author:        Kaiming Zhu
@Date:          2023/8/9 2:38
"""

import os


def ensure_make_path(path: str):
    assert (is_legal_path(path), f"Path {path} is illegal")
    if not os.path.exists(path):
        os.makedirs(path)


def is_path_exist(path: str) -> bool:
    return is_legal_path(path) and os.path.exists(path)


def is_legal_path(path: str) -> bool:
    return os.path.isdir(path) or os.path.isfile(path)
