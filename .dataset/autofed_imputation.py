"""
@Encoding:      UTF-8
@File:          autofed_imputation.py

@Introduction:  impute modal incomplete dataset with AutoFed Encoder
@Author:        Kaiming Zhu
@Date:          2023/12/22 10:43
@Reference:     "autoencoder4autofed_pytorch.py" via Songcan Yu.
                https://stackoverflow.com/questions/44522863/calculate-pca-using-numpy
"""

from copy import deepcopy
from functools import reduce
from operator import add
import os
import random
import sys
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import scipy.io as sio

sys.path.append("..")
from fldataset.converter.torch import MultiModalDataset
from simulator.utils.aggregator import copy_state_dict

# hyper-param: where to load original dataset
dataset_path = "./HAR/incompleteness/"
# hyper-param: where to archive incompleteness dataset
result_path = "./HAR/imputation/"
# hyper-param: keys of each modal data
modal_keys = ["acc_feats", "gyro_feats"]

# hyper-param: epoch for training encoder
epoch = 1000
# hyper-param: learning rate
lr = 3e-4
# hyper-param: batch size for training encoder
batch_size = 32
# hyper-param: optimizer for training encoder
optimizer = optim.Adam
# hyper-param: criterion for evaluation
criterion = nn.MSELoss
# hyper-param: device to train datas, set it as 'cpu' if run without GPU supporting.
device = "cuda:0"
# hyper-param: maximum epoch that model can have a downgraded performance, and it will make early-stop if over it.
max_degrade_epoch = round(epoch * 0.04)
# hyper-param: is need to save the best encoder checkpoint, will be named as f'client{i}.encoder.pt' for each client.
need_saving_best_encoder = False
# hyper-param: specified random seed, set it as 'None' if you don't need it
seed: Optional[int] = None

# fixed random seed
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)


class Encoder(nn.Module):
    def __init__(self, data_shape: Tuple[int]):
        super().__init__()
        dimension = data_shape[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_features=dimension, out_features=dimension, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=dimension, out_features=dimension // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=dimension // 2, out_features=dimension // 2, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=dimension // 2, out_features=dimension),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    filenames = list(os.listdir(dataset_path))
    filenames = [filename for filename in filenames if filename.endswith(".mat")]
    filenames.sort()
    for user_index, filename in enumerate(filenames):
        print(f"User {user_index}: Start To Train")
        data_by_name = sio.loadmat(dataset_path + filename, squeeze_me=True)
        data_by_name = deepcopy(data_by_name)
        datas = [data_by_name[key] for key in modal_keys]
        labels = data_by_name["labels"]

        # Distinguish datas and labels into two categories: 'complete' and 'incomplete'.
        complete_indices = []
        incomplete_indices = []
        for index in range(0, labels.shape[0]):
            current_index_datas = [modal_data[index] for modal_data in datas]
            is_each_modal_complete = [np.any(modal_data != 0) for modal_data in current_index_datas]
            if all(is_each_modal_complete):
                complete_indices.append(index)
            else:
                incomplete_indices.append(index)

        complete_datas = [modal_data[complete_indices] for modal_data in datas]
        complete_labels = labels[complete_indices]
        incomplete_datas = [modal_data[incomplete_indices] for modal_data in datas]
        incomplete_labels = labels[incomplete_indices]

        # Train Encoder, shuffle data in each epoch
        dimension = complete_datas[0].shape[-1]
        model = Encoder(data_shape=(dimension,)).to(device)
        opt = optimizer(model.parameters(), lr=lr)
        loss_fn = criterion()

        dataset = MultiModalDataset(
            modal_data_by_key={key: modal_data for key, modal_data in zip(modal_keys, complete_datas)},
            targets=complete_labels
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        # Define all utils and send gpu first, avoiding frequently I/O.
        degrade_epoch = torch.tensor(0.0).to(device)
        current_loss = torch.tensor(0.0).to(device)
        best_loss = torch.tensor(float('nan')).to(device)
        best_state_dict = None
        for current_epoch in range(0, epoch):
            # Train Encoder
            model.train()
            for data_by_key in data_loader:
                batch_datas = [data_by_key[key].to(device) for key in dataset.data_keys]
                outputs = [model(modal_data) for modal_data in batch_datas]
                losses = [loss_fn(origin_data, output) for origin_data, output in zip(batch_datas, outputs)]
                loss = reduce(add, losses)

                opt.zero_grad()
                loss.backward()
                opt.step()

            # Record Best Encoder
            model.eval()
            current_loss.zero_()
            for data_by_key in data_loader:
                batch_datas = [data_by_key[key].to(device) for key in dataset.data_keys]
                outputs = [model(modal_data) for modal_data in batch_datas]
                losses = [loss_fn(origin_data, output) for origin_data, output in zip(batch_datas, outputs)]
                loss = reduce(add, losses)
                current_loss += loss

            current_loss /= len(dataset)
            print(f"Epoch: {current_epoch}, loss: {str(current_loss.item()).format('.5f')}")

            if (current_loss <= best_loss).item() or torch.isnan(best_loss).item():
                best_loss = current_loss.clone().detach()
                best_state_dict = copy_state_dict(model.state_dict())
                degrade_epoch.zero_()
            else:
                degrade_epoch += 1
                if degrade_epoch.item() > max_degrade_epoch:
                    break

        # imputation on modal incomplete datas
        imputation_datas = [[] for _ in incomplete_datas]
        model.load_state_dict(best_state_dict)
        incomplete_dataset = MultiModalDataset(
            modal_data_by_key={key: modal_data for key, modal_data in zip(modal_keys, incomplete_datas)},
            targets=incomplete_labels
        )
        incomplete_data_loader = DataLoader(
            dataset=incomplete_dataset,
            batch_size=min(batch_size, len(incomplete_dataset)),
            num_workers=0,
            pin_memory=False
        )
        for data_by_key in incomplete_data_loader:
            batch_datas = [data_by_key[key].to(device) for key in dataset.data_keys]
            outputs = [model(modal_data) for modal_data in batch_datas]
            for result, output in zip(imputation_datas, outputs):
                result.append(output.detach().cpu().numpy())
        imputation_datas = [np.concatenate(datas) for datas in imputation_datas]

        # rearrange datas and labels, w.r.t. complete_datas and imputation_datas
        result_datas = [[] for _ in complete_datas]
        result_labels = []
        while len(complete_indices) > 0 or len(incomplete_indices) > 0:
            if len(complete_indices) == 0:
                for result, modal_data in zip(result_datas, imputation_datas):
                    result.append(modal_data)
                result_labels.append(incomplete_labels)
                break
            if len(incomplete_indices) == 0:
                for result, modal_data in zip(result_datas, complete_datas):
                    result.append(modal_data)
                result_labels.append(complete_labels)
                break

            if complete_indices[0] <= incomplete_indices[0]:
                complete_indices = complete_indices[1:]
                for i, (result, modal_data) in enumerate(zip(result_datas, complete_datas)):
                    result.append(modal_data[0])
                    complete_datas[i] = modal_data[1:]

                result_labels.append(complete_labels[0])
                complete_labels = complete_labels[1:]
            else:
                incomplete_indices = incomplete_indices[1:]
                for i, (result, modal_data) in enumerate(zip(result_datas, imputation_datas)):
                    result.append(modal_data[0])
                    imputation_datas[i] = modal_data[1:]

                result_labels.append(incomplete_labels[0])
                incomplete_labels = incomplete_labels[1:]

        # reshape datas, labels to expected one
        for i, result_modal_data in enumerate(result_datas):
            target_shape = (1, ) + result_modal_data[-1].shape[1:]
            for j, data in enumerate(result_modal_data[0: -1]):
                data = data.reshape(target_shape)
                result_modal_data[j] = data
            result_datas[i] = np.concatenate(result_modal_data)

        target_shape = (1, ) + result_labels[-1].shape[1:]
        for i, label in enumerate(result_labels[0: -1]):
            label = label.reshape(target_shape)
            result_labels[i] = label
        result_labels = np.concatenate(result_labels)

        # write result to new datasets
        for key, modal_data in zip(modal_keys, result_datas):
            data_by_name[key] = modal_data
        data_by_name["labels"] = result_labels
        data_by_name["autofed_imputation_result_loss"] = str(best_loss.item()).format('.5f')
        sio.savemat(result_path + filename, data_by_name)

        if need_saving_best_encoder:
            torch.save(best_state_dict, result_path + filename + ".encoder.pt")

        print(f"User {user_index}: Finish Imputation with loss {str(best_loss.item()).format('.5f')}")
