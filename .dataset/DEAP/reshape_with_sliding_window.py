"""
@Encoding:      UTF-8
@File:          reshape_with_sliding_window.py

@Introduction:  Reshape DEAP Dataset via sliding window
@Author:        Kaiming Zhu
@Date:          2023/12/10 2:06
"""

from copy import deepcopy
import os

import scipy.io as sio
import numpy as np

# hyper-param: where to load original dataset
dataset_path = "./incomplete/"
# hyper-param: where to archive window_slided dataset
result_path = "./window_slided/"
# hyper-param: keys of each modal
modal_keys = ["eeg_feats", "emg_feats", "eog_feats"]
# hyper-param: size of sliding window
window_size = 3


if not os.path.exists(result_path):
    os.makedirs(result_path)


filenames = list(os.listdir(dataset_path))
filenames = [filename for filename in filenames if filename.endswith(".mat")]
filenames.sort()
for filename in filenames:
    # Step 1. Load Dataset, store at 'modal_datas' and 'labels'
    data_by_name = sio.loadmat(dataset_path + filename, squeeze_me=True)
    data_by_name = deepcopy(data_by_name)

    def make_sliding_window_sampling_by(data_keys, label_key):
        modal_datas: [np.ndarray] = [data_by_name[key] for key in data_keys]
        labels: np.ndarray = data_by_name[label_key]

        # Step 2. Make Interation on Datas, which is formatted as (Experiment, Time Series, Features).
        result_modal_datas = [[] for _ in modal_datas]
        result_labels = []
        experiment_time = modal_datas[0].shape[0]
        for time in range(0, experiment_time):
            # Step 3. Retrieve label in this experiment, and get data of each modal in this experiment
            current_label = labels[time]
            current_modal_datas = [modal_data[time] for modal_data in modal_datas]
            time_series_amount = current_modal_datas[0].shape[0]
            if time_series_amount < window_size:
                continue

            # Step 4. sample with sliding windows
            sample_indices = list(range(0, time_series_amount - window_size))
            for modal_index, modal_data in enumerate(current_modal_datas):
                sample_datas = [
                    modal_data[i: i + window_size]
                    for i in sample_indices
                ]
                result_modal_datas[modal_index].extend(sample_datas)

            # Step 5. add same amount of labels to result
            result_labels.extend([current_label for _ in sample_indices])

        return result_modal_datas, result_labels

    # Step 6. update dict
    data_keys = [key for key in modal_keys]
    label_key = "labels"
    modal_datas, labels = make_sliding_window_sampling_by(data_keys=data_keys, label_key=label_key)
    data_by_name[label_key] = np.array(labels)
    for i, (key, modal_data) in enumerate(zip(data_keys, modal_datas)):
        data_by_name[key] = np.array(modal_data)

    # Step 7. save it to result folder
    sio.savemat(result_path + filename, data_by_name)
