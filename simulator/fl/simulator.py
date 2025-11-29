"""
@Encoding:      UTF-8
@File:          simulator.py

@Introduction:  Definition of fl.Simulator
@Author:        Kaiming Zhu
@Date:          2023/8/10 21:35
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

from . import Client
from ..core import DLSimulator
from ..utils.aggregator import fedavg


class Simulator(DLSimulator):
    """ Federated Learning Simulator """
    def __init__(
        self,
        clients: [Client],
        global_model: nn.Module,
        global_epochs: int,
        archive_folder_name: Optional[str] = None,
        device: str = "cpu"
    ):
        super().__init__(
            clients=clients,
            global_model=global_model,
            global_epochs=global_epochs,
            archive_folder_name=archive_folder_name,
            device=device
        )

    def _boardcast_parameters(self):
        for client in self.clients:
            client.update_with_state_dict(self.global_model.state_dict())

    def _local_training(self):
        for i, client in enumerate(self.clients):
            print(f"Client {i} start training")
            client.train()

    def _aggregation(self):
        weights = self.trainset_proportions()
        weights = torch.from_numpy(np.array(weights))
        weights.to(self.device)

        new_state_dict = fedavg(
            param_names=list(self.global_model.state_dict().keys()),
            weights=weights,
            state_dicts=[client.state_dict() for client in self.clients]
        )
        self.global_model.load_state_dict(new_state_dict)

    def _global_epoch_will_start(self, epoch: int):
        self._visualize_training_metrics_with_tracking()

    def _simulation_will_finish(self):
        self._visualize_training_metrics_with_tracking()

    def _visualize_training_metrics_with_tracking(self):
        metrics_by_key = self.eval()
        test_acc, test_loss = metrics_by_key["test_acc"], metrics_by_key["test_loss"]
        print(f"global epoch {self.current_epoch}: test acc {test_acc}, test loss {test_loss}")

        self.archive_metrics(epoch=self.current_epoch, metrics_dict=metrics_by_key)
