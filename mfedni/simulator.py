"""
@Encoding:      UTF-8
@File:          simulator.py

@Introduction:  Definition of mfedni.Simulator
@Author:        Kaiming Zhu, Songcan Yu, Feiyuan Liang.
@Date:          2023/12/27 21:52
"""

from copy import deepcopy
import time
from typing import Optional, Dict, List

import numpy as np
import random
import torch
from torch import nn

from . import Client
from simulator.core import DLSimulator
from simulator.utils.aggregator import fedavg
from simulator.utils.model import copy_state_dict


class Simulator(DLSimulator):
    """ Federated Learning Simulator """
    def __init__(
        self,
        clients: [Client],
        global_model: nn.Module,
        global_epochs: int,
        archive_folder_name: Optional[str] = None,
        device: str = "cpu",
        each_round_client_amount: Optional[int] = None,
        need_client_shuffle: bool = False,
    ):
        super().__init__(
            clients=clients,
            global_model=global_model,
            global_epochs=global_epochs,
            archive_folder_name=archive_folder_name,
            device=device
        )
        # prototype calculation: core properties
        self._client_models = []
        self._global_prototype_by_label: Dict[int, torch.Tensor] = {}

        # prototype calculation: caching properties, to accelerate calculation
        self._all_possible_labels = []

        # partial clients selection
        self.each_round_client_amount = each_round_client_amount
        self.need_client_shuffle = need_client_shuffle
        self._this_round_clients = []

        # record best model
        self.__best_acc: float = 0.0
        self.__best_state_dict: Optional[dict] = None
        
        self._time_stamp = time.time()

    def clients_for_this_round(self, need_refresh: bool = False):
        if self.each_round_client_amount is None:
            return self.clients

        if need_refresh:
            client_indices = list(range(0, len(self.clients)))
            if self.need_client_shuffle:
                random.shuffle(client_indices)

            client_indices = client_indices[0: self.each_round_client_amount]
            self._this_round_clients = []
            for index in client_indices:
                self._this_round_clients.append(self.clients[index])

        return self._this_round_clients

    def _boardcast_parameters(self):
        # biz logic
        for client in self.clients:
            client.global_prototypes_by_label = self._global_prototype_by_label
            client.update_with_state_dict(self.global_model.state_dict())

        # visualization
        self._visualize_training_metrics_with_tracking()

    def _local_training(self):
        for client in self.clients_for_this_round(need_refresh=True):
            client.train()

    def _aggregation(self):
        # fedavg
        trainset_proportions = self.trainset_proportions(clients=self.clients_for_this_round())
        trainset_proportions = torch.from_numpy(np.array(trainset_proportions))
        trainset_proportions.to(self.device)

        new_state_dict = fedavg(
            param_names=list(self.global_model.state_dict().keys()),
            weights=trainset_proportions,
            state_dicts=[client.state_dict() for client in self.clients_for_this_round()]
        )
        self.global_model.load_state_dict(new_state_dict)

        # update global prototypes
        self._global_prototype_by_label.clear()
        each_client_prototype_by_label = [client.prototype_by_label for client in self.clients_for_this_round()]
        if len(self._all_possible_labels) == 0:
            all_possible_labels = []
            for prototype_by_label in each_client_prototype_by_label:
                labels = list(prototype_by_label.keys())
                all_possible_labels.extend(labels)
            self._all_possible_labels = list(set(all_possible_labels))

        for label in self._all_possible_labels:
            client_indices: [int] = []
            prototypes: [torch.Tensor] = []
            for index, prototype_by_label in enumerate(each_client_prototype_by_label):
                if label not in prototype_by_label.keys():
                    continue
                client_indices.append(index)
                prototypes.append(prototype_by_label[label])

            weights = [trainset_proportions[index] for index in client_indices]
            weights = [weight / sum(weights) for weight in weights]

            prototype_for_label = weights[0] * prototypes[0]
            for weight, prototype in zip(weights[1:], prototypes[1:]):
                prototype_for_label += weight * prototype

            self._global_prototype_by_label[label] = prototype_for_label

    def _simulation_will_start(self):
        self.__best_acc = 0.0
        self.__best_state_dict = None

    def _simulation_will_finish(self):
        self._boardcast_parameters()
        self._visualize_training_metrics_with_tracking()

        if self.__best_state_dict is not None:
            folder = self.archive_path
            filename = "best_model.pt"
            torch.save(self.__best_state_dict, folder + filename)

    def _visualize_training_metrics_with_tracking(self):
        # save current acc and loss
        metrics_by_key = self.eval()
        test_acc, test_loss = metrics_by_key["test_acc"], metrics_by_key["test_loss"]
        train_acc, train_loss = metrics_by_key["train_acc"], metrics_by_key["train_loss"]
        
        time_stamp_now = time.time()
        if self.current_epoch == 0:
            print(f"global epoch {self.current_epoch}/{self.global_epoch}: ", end="")
            print(f"train acc {'{:.2f}%'.format(train_acc * 100)}, train loss {'{:.4f}'.format(train_loss)}, ", end="")
            print(f"test acc {'{:.2f}%'.format(test_acc * 100)}, test loss {'{:.4f}'.format(test_loss)}.")
        else:
            print(f"global epoch {self.current_epoch}/{self.global_epoch} [{time_stamp_now - self._time_stamp:4.2f}s]: ", end="")
            print(f"train acc {'{:.2f}%'.format(train_acc * 100)}, train loss {'{:.4f}'.format(train_loss)}; ", end="")
            print(f"test acc {'{:.2f}%'.format(test_acc * 100)}, test loss {'{:.4f}'.format(test_loss)}.")
        
        self._time_stamp = time_stamp_now
        self.archive_metrics(epoch=self.current_epoch, metrics_dict=metrics_by_key)

        # record for best acc and state dict
        if test_acc > self.__best_acc:
            self.__best_acc = test_acc
            self.__best_state_dict = copy_state_dict(self.global_model.state_dict())
