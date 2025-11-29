"""
@Encoding:      UTF-8
@File:          pca_reduction.py

@Introduction:  PCA reduction for Dataset
@Author:        Kaiming Zhu
@Date:          2023/12/25 17:19
@Reference:     https://www.kdnuggets.com/2023/05/principal-component-analysis-pca-scikitlearn.html
"""

from copy import deepcopy
import os
import random
import sys

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

sys.path.append("..")

# hyper-param: where to load original dataset
dataset_path = "./HAR/origin/"
# hyper-param: where to archive incompleteness dataset
result_path = "./HAR/pca_reduction/"
# hyper-param: keys of each modal data
modal_keys = ["acc_feats", "gyro_feats"]
# hyper-param: expected dimension for PCA
# Notes: target_dimension should in range with (0, min(n_samples, n_dimensions))
target_dimension = 128
# hyper-param: specified random seed, set it as 'None' if you do not need it
seed = None

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)


if not os.path.exists(result_path):
    os.makedirs(result_path)

filenames = list(os.listdir(dataset_path))
filenames = [filename for filename in filenames if filename.endswith(".mat")]
filenames.sort()
for filename in filenames:
    data_by_name = sio.loadmat(dataset_path + filename, squeeze_me=True)
    data_by_name = deepcopy(data_by_name)
    datas = [data_by_name[key] for key in modal_keys]

    pca = PCA(n_components=target_dimension)
    datas = [pca.fit_transform(modal_data) for modal_data in datas]

    for i, (key, modal_data) in enumerate(zip(modal_keys, datas)):
        data_by_name[key] = np.array(modal_data)
    data_by_name["is_pca_reducted"] = "True"
    sio.savemat(result_path + filename, data_by_name)
