"""
@Encoding:      UTF-8
@File:          reshape_with_sliding_window.py

@Introduction:  Reshape HAR Dataset via sliding window
@Author:        Kaiming Zhu
@Date:          2023/12/9 22:37
@Reference:     https://en.wikipedia.org/wiki/Sliding_window_protocol
                https://stackoverflow.com/questions/4588628/find-indices-of-elements-equal-to-zero-in-a-numpy-array
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
modal_keys = ["acc_feats", "gyro_feats"]
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

        # Step 2. Calculate new shape for data in each modal
        modal_data_shapes = [modal_data[0].shape for modal_data in modal_datas]
        for i, shape in enumerate(modal_data_shapes):
            new_shape = list(shape)
            new_shape.insert(-2, window_size)
            modal_data_shapes[i] = tuple(new_shape)

        result_modal_datas = [[] for _ in modal_datas]
        result_labels = []
        while len(labels) > 0:
            # Step 3. Make Iteration on Labels, find first index which is not continuous,
            #         and store at first_not_continuous_index.
            current_label = labels[0]
            not_continuous_indices = np.where(labels != current_label)[0]
            if len(not_continuous_indices) == 0:
                first_not_continuous_index = len(labels)
            else:
                first_not_continuous_index = not_continuous_indices[0]

            # Step 4. Iterate all modals, and make sliding, reshaping
            current_indices = list(range(0, first_not_continuous_index))
            for modal_index, modal_data in enumerate(modal_datas):
                # make window sliding w.r.t. window size.
                current_modal_data = modal_data[current_indices]
                window_slided_datas = []
                for i in current_indices[0: -1 * window_size + 1]:
                    sample_datas = []
                    for j in range(i, i + window_size):
                        sample_datas.append(current_modal_data[j])
                    window_slided_datas.append(sample_datas)

                # reshape to expected size.
                window_slided_datas = [np.hstack(data) for data in window_slided_datas]
                window_slided_datas = [data.reshape(modal_data_shapes[modal_index]) for data in window_slided_datas]

                # store at 'result_modal_datas'
                result_modal_datas[modal_index].extend(window_slided_datas)

            # Step 5. Store labels
            result_labels.extend([current_label for _ in current_indices[0: -1 * window_size + 1]])

            # Step 6. truncate labels, datas w.r.t. first_not_continuous_index
            labels = labels[first_not_continuous_index:]
            modal_datas = [modal_data[first_not_continuous_index:] for modal_data in modal_datas]

        return result_modal_datas, result_labels

    # Step 8. update dict
    data_keys = [key for key in modal_keys]
    label_key = "labels"
    modal_datas, labels = make_sliding_window_sampling_by(data_keys=data_keys, label_key=label_key)
    data_by_name[label_key] = labels
    for i, (key, modal_data) in enumerate(zip(data_keys, modal_datas)):
        data_by_name[key] = np.array(modal_data)

    # Step 9. save it to result folder
    sio.savemat(result_path + filename, data_by_name)
