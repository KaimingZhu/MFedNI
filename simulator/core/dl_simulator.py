"""
@Encoding:      UTF-8
@File:          dlsimulator.py

@Introduction:  definition of `flsimulator.DLSimulator`
@Author:        Kaiming Zhu
@Date:          2023/8/7 21:29
"""

from __future__ import annotations

import os
from typing import Optional, Any, Dict, List
import re

import numpy as np
import torch
import torch.nn as nn

from .client import Client
from ..utils.path import ensure_make_path, is_path_exist


class DLSimulator:
    """Base Class, Distributed Learning(DL) Simulator, providing common method and callback."""
    archive_rootpath = "./.result/"

    def __init__(
        self,
        clients: [Client],
        global_model: nn.Module,
        global_epochs: int,
        archive_folder_name: Optional[str] = None,
        device: str = "cpu",
        need_checkpoint_archiving: bool = True,
    ):
        # dl training
        self._clients: [Client] = clients
        self.global_model: nn.Module = global_model
        self.global_epoch: int = global_epochs

        # computing properties
        self.device = device
        self._current_epoch = 0
        self.archive_path = None
        self.need_checkpoint_archiving = need_checkpoint_archiving
        if archive_folder_name is not None:
            self.archive_path = DLSimulator.simulation_archive_path(folder_name=archive_folder_name)

        # status flag
        self._is_running = False

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, epoch):
        self._current_epoch = epoch
        if self._is_running:
            self._sync_current_epoch_to_clients()

    @property
    def clients(self) -> [Client]:
        return self._clients

    @clients.setter
    def clients(self, clients: [Client]):
        old_value = self._clients
        self._clients = clients
        self._clients_did_change(old_value=old_value)

    def _clients_did_change(self, old_value: [Client]):
        """ Event invoking for subclasses, for clients in Simulator has changed.
        Args:
            old_value([Client]): old value of clients.
        Returns:
            None
        """
        pass

    def is_running(self) -> bool:
        return self._is_running

    def _need_archiving(self) -> bool:
        return self.archive_path is not None

    def _sync_properties_to_clients(self):
        self._sync_archving_path_to_clients()
        self._sync_device_to_clients()
        self._sync_current_epoch_to_clients()

    def _sync_archving_path_to_clients(self):
        if not self._need_archiving():
            for client in self.clients:
                client.archive_path = None
            return

        ensure_make_path(self.archive_path)
        for index, client in enumerate(self.clients):
            client.archive_path = self.archive_path + f"client_{index}/"
            ensure_make_path(client.archive_path)

    def _sync_device_to_clients(self):
        for client in self.clients:
            client.device = self.device

    def _sync_current_epoch_to_clients(self):
        for client in self.clients:
            client.current_global_epoch = self.current_epoch

    def eval(self, model: nn.Module = None) -> Dict:
        """eval model with data holding by each client, return the result in dict.
        Args:
            model(nn.Module): model you need to eval. If this value is `None`,
            client will return the eval result of its own model. Default: `None`.

        Returns:
            a Dict[str, Any] object, which contains eight keys:
            - "train_acc": value type is `float`, means the average accuracy on trainset, ranging 0 to 1.
            - "test_loss": value type is `float`, means the average loss on testset.
            - "test_acc": value type is `float`, means the average accuracy on testset, ranging 0 to 1.
            - "test_loss": value type is `float`, means the average loss on testset.
            - "train_acc_list": value type is `List[float]`, means the average accuracy of each clients' trainset.
            - "train_loss_list": value type is `List[float]`, means the average loss of clients' trainset.
            - "test_acc_list": value type is `List[float]`, means the average accuracy of each clients' testset.
            - "test_loss_list": value type is `List[float]`, means the average loss of clients' testset.
        """
        eval_model: nn.Module = self.global_model
        if model is not None:
            eval_model = model
            eval_model.to(self.device)

        destination: Dict[str: Any] = {}

        train_acc_list: [float] = []
        train_loss_list: [float] = []
        test_acc_list: [float] = []
        test_loss_list: [float] = []
        for client in self.clients:
            metrics_by_key = client.eval(eval_model)
            train_acc_list.append(metrics_by_key["train_acc"])
            train_loss_list.append(metrics_by_key["train_loss"])
            test_acc_list.append(metrics_by_key["test_acc"])
            test_loss_list.append(metrics_by_key["test_loss"])
        destination["train_acc_list"] = train_acc_list
        destination["train_loss_list"] = train_loss_list
        destination["test_acc_list"] = test_acc_list
        destination["test_loss_list"] = test_loss_list

        trainset_proportions = self.trainset_proportions()
        testset_proportions = self.testset_proportions()
        train_acc = np.sum(np.array(train_acc_list) * np.array(trainset_proportions)).tolist()
        train_loss = np.sum(np.array(train_loss_list) * np.array(trainset_proportions)).tolist()
        test_acc = np.sum(np.array(test_acc_list) * np.array(testset_proportions)).tolist()
        test_loss = np.sum(np.array(test_loss_list) * np.array(testset_proportions)).tolist()

        destination["train_acc"] = train_acc
        destination["train_loss"] = train_loss
        destination["test_acc"] = test_acc
        destination["test_loss"] = test_loss

        return destination

    def trainset_proportions(self, clients: Optional[List[Client]] = None) -> List[float]:
        """ return the ratio of clients' trainset to overall. """
        if clients is None:
            clients = self.clients

        lengths = [client.trainset_length() for client in clients]
        return (np.array(lengths) / np.sum(lengths)).tolist()

    def testset_proportions(self, clients: Optional[List[Client]] = None) -> List[float]:
        """ return the ratio of clients' testset to overall. """
        if clients is None:
            clients = self.clients

        lengths = [client.testset_length() for client in clients]
        return (np.array(lengths) / np.sum(lengths)).tolist()

    def run(self):
        """ function to perform DL simulation. """
        if self.current_epoch >= self.global_epoch:
            return

        self._is_running = True
        self._sync_properties_to_clients()
        self._simulation_will_start()
        for _ in range(self.current_epoch, self.global_epoch):
            # Event Invoking
            self._archive_training_checkpoint_if_needed()
            self._global_epoch_will_start(epoch=self.current_epoch)

            # Algorithm
            self._boardcast_parameters()
            self._local_training()
            self._aggregation()

            self.current_epoch += 1

        # Event Invoking
        self._archive_training_checkpoint_if_needed()
        self._simulation_will_finish()
        self._is_running = False

    def _boardcast_parameters(self):
        pass

    def _local_training(self):
        pass

    def _aggregation(self):
        pass

    def _global_epoch_will_start(self, epoch: int):
        """ Callback for global epoch begin. """
        pass

    def _simulation_will_start(self):
        pass

    def _simulation_will_finish(self):
        """ Callback for simulation finished. """
        pass

    def recover_training_checkpoint(self) -> Optional[int]:
        """try to recover DL simulation by pre-archived checkpoints.
        Returns:
            An object means recovered_epoch(Optional, int):
                - `None` means that Simulator can not recover to any epoch.
                - Integer means the max global epoch that simulator has been recovered to.
        """
        if not is_path_exist(self.archive_path):
            return None

        available_epochs: List[int] = self._filter_available_epochs(path=self.archive_path)
        if len(available_epochs) == 0:
            return None

        chosen_epoch = self._choose_max_epoch_can_recover(available_epochs=available_epochs)
        if chosen_epoch is None:
            return None

        checkpoint_filename = self._checkpoint_filename(global_epoch=chosen_epoch)
        checkpoint = torch.load(self.archive_path + checkpoint_filename)
        self.global_model.load_state_dict(checkpoint["model"])
        for i, client in enumerate(self.clients):
            client.recover_training_checkpoint(global_epoch=chosen_epoch)

        self.current_epoch = chosen_epoch
        return chosen_epoch

    def _archive_training_checkpoint_if_needed(self):
        """ archive necessary information if needed, to support Distributed Learning recovering from given epoch. """
        if not (self._need_archiving() and self.need_checkpoint_archiving):
            return

        checkpoint = {"model": self.global_model.state_dict()}
        checkpoint_filename = self._checkpoint_filename()
        torch.save(checkpoint, self.archive_path + checkpoint_filename)

        for client in self.clients:
            client.archive_training_checkpoint()

    @staticmethod
    def _filter_available_epochs(path: str) -> List[int]:
        files = os.listdir(path)
        if len(files) == 0:
            return []

        def is_check_point_file(file_name: str) -> bool:
            return re.match("^checkpoint_[0-9]+.pt$", file_name) is not None
        checkpoint_files = list(filter(is_check_point_file, files))

        available_epochs = [name[11:] for name in checkpoint_files]
        available_epochs = [int(name[:-3]) for name in available_epochs]
        return available_epochs

    def _choose_max_epoch_can_recover(self, available_epochs) -> Optional[int]:
        self._sync_archving_path_to_clients()

        available_epochs.sort(reverse=True)
        for epoch in available_epochs:
            has_client_archived_list = [
                client.has_archived_training_checkpoint(global_epoch=epoch)
                for client in self.clients
            ]
            if all(has_client_archived_list):
                return epoch

        return None

    @classmethod
    def client_archive_path(cls, folder_name: str, client_index: int) -> str:
        return cls.simulation_archive_path(folder_name=folder_name) + f"client_{client_index}/"

    @classmethod
    def simulation_archive_path(cls, folder_name: str) -> str:
        return cls.archive_rootpath + folder_name + "/"

    def archive_metrics(self, epoch: int, metrics_dict: Dict):
        if not self._need_archiving():
            return
        
        folder = self.archive_path
        filename = self._metrics_archive_filename()
        ensure_make_path(folder)

        metrics_by_global_epoch = {}
        if is_path_exist(folder + filename):
            metrics_by_global_epoch = torch.load(folder + filename)

        metrics_in_this_epoch = metrics_dict
        if epoch in metrics_by_global_epoch.keys():
            archived_metrics = metrics_by_global_epoch[epoch]
            if isinstance(archived_metrics, Dict):
                archived_metrics.update(metrics_dict)
                metrics_in_this_epoch = archived_metrics

        metrics_by_global_epoch[epoch] = metrics_in_this_epoch
        torch.save(metrics_by_global_epoch, folder + filename)

    def load_metrics(self) -> Dict:
        folder = self.archive_path
        filename = self._metrics_archive_filename()
        if not is_path_exist(folder + filename):
            return {}
        else:
            return torch.load(folder + filename)

    def _metrics_archive_filename(self):
        return "metrics.pt"

    def _checkpoint_filename(self, global_epoch: Optional[int] = None):
        if global_epoch is None:
            global_epoch = self.current_epoch
        return f"checkpoint_{global_epoch}.pt"
