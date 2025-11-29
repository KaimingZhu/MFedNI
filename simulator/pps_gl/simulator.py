"""
@Encoding:      UTF-8
@File:          simulator.py

@Introduction:  Definition of pps_gl.Simulator
@Author:        Kaiming Zhu
@Date:          2023/8/17 21:25
"""

from typing import Optional, Dict

import numpy as np
import torch
from torch import nn

from simulator.core import Client, DLSimulator
import simulator.pps_gl as pps_gl
from ..utils.aggregator import gradient_based_fedavg


class Simulator(DLSimulator):
    """ Privacy-Preserving Gossip Learning Simulator. """

    def __init__(
        self,
        clients: [Client],
        global_model: nn.Module,
        global_epochs: int,
        guard_client_index: int,
        archive_folder_name: Optional[str] = None,
        device: str = "cpu",
        clients_select_ratio: float = 1.0,
        subtask_divide_ratio: float = 0.2,
        local_lr: Optional[float] = 0.005,
        local_lr_decay_coefficient: Optional[float] = 0.999,
    ):
        super().__init__(
            clients=clients,
            global_model=global_model,
            global_epochs=global_epochs,
            archive_folder_name=archive_folder_name,
            device=device
        )
        # Properties for GL Subtask Grouping
        self.clients_select_ratio = clients_select_ratio
        self.subtask_divide_ratio = subtask_divide_ratio
        self.client_subtask_groups: [[Client]] = []
        self._client_id_by_client: Dict[Client, int] = {client: i for i, client in enumerate(self._clients)}

        # Properties for GL local training
        self.local_lr_decay_coefficient: Optional[float] = local_lr_decay_coefficient
        self.local_lr: Optional[float] = local_lr
        if self.local_lr is not None:
            for client in self.clients:
                client.lr = self.local_lr

        # Properties for privacy preserving
        self.guard_client_index = guard_client_index

    def _guard_client(self) -> Client:
        return self.clients[self.guard_client_index]

    def _private_clients(self) -> [Client]:
        return self.clients[0:self.guard_client_index] + self.clients[self.guard_client_index+1:]

    def _clients_did_change(self, old_value: [Client]):
        self._client_id_by_client = {client: i for i, client in enumerate(self.clients)}

    def _global_epoch_will_start(self, epoch: int):
        # Algorithm: Divide private clients into several groups
        selected_private_client_index: [int] = self._generate_random_selected_index(ratio=self.clients_select_ratio)
        client_index_groups: [[int]] = self._make_groupped_indexs(
            indexs=selected_private_client_index,
            grouping_ratio=self.subtask_divide_ratio
        )

        clients = self.clients
        self.client_subtask_groups = []
        for group in client_index_groups:
            if len(group) == 0:
                continue
            clients_group: [Client] = [clients[index] for index in group]
            self.client_subtask_groups.append(clients_group)

        # Visualization
        metrics_dict = self.eval()
        test_acc, test_loss = metrics_dict["test_acc"], metrics_dict["test_loss"]
        print(f"global Epoch {epoch}: test acc {test_acc}, test loss {test_loss}")
        print("clients group results:")
        for i, group in enumerate(client_index_groups):
            print(f"group {i}: {group}")

        # Track
        metrics_dict.update({"guard_client_index": self.guard_client_index, "subtask_groups": client_index_groups})
        self.archive_metrics(epoch=self.current_epoch, metrics_dict=metrics_dict)

    def _generate_random_selected_index(self, ratio: float) -> [int]:
        if ratio <= 0:
            return []

        indexes = np.arange(0, len(self.clients))
        private_client_indexes = list(filter(lambda index: index != self.guard_client_index, indexes))
        np.random.shuffle(private_client_indexes)
        if ratio >= 1.0:
            return private_client_indexes
        else:
            select_slice = slice(0, int(ratio * len(private_client_indexes)))
            return private_client_indexes[select_slice]

    def _make_groupped_indexs(self, indexs: [int], grouping_ratio: float) -> [[int]]:
        if grouping_ratio < 0 or grouping_ratio > 1:
            return [[] for _ in range(0, len(indexs))]
        elif grouping_ratio > 1:
            return [[indexs]]

        grouping_start_index = (np.arange(0, 1.0, grouping_ratio) * len(indexs)).astype('int64')
        index_groups: [[Client]] = []
        for (i, index) in enumerate(grouping_start_index):
            if i != len(grouping_start_index) - 1:
                group_slice = slice(index, grouping_start_index[i + 1])
                index_groups.append(indexs[group_slice])
            else:
                index_groups.append(indexs[index:])

        return index_groups

    def _boardcast_parameters(self):
        # Algorithm
        self._guard_client().update_with_state_dict(state_dict=self.global_model.state_dict())
        for client_group in self.client_subtask_groups:
            first_client = client_group[0]
            first_client.update_with_state_dict(state_dict=self.global_model.state_dict())

        # Event Invoking
        for client in self.clients:
            if not isinstance(client, pps_gl.Client):
                continue
            client._parameter_boardcast_did_finish()

    def _local_training(self):
        guard_client = self._guard_client()
        print(f"guard client: client {self.guard_client_index} begin training.")
        guard_client.train()
        guard_client_gradient = self._guard_client().gradient()

        for group_index, group in enumerate(self.client_subtask_groups):
            # Visualization
            print(f"Group {group_index} begin training.")
            for (i, client) in enumerate(group):
                # Visualization
                print(f"client {self._client_id_by_client[client]} begin training")

                # Algorithm
                self.update_client_lr_if_needed(client=client, group_index=i)
                if i == 0:
                    client.update_with_gradient(gradient=guard_client_gradient)
                else:
                    last_client: Client = group[i - 1]
                    client.update_with_gradient(gradient=last_client.gradient())
                client.train()

    def update_client_lr_if_needed(self, client: Client, group_index: int):
        if self.local_lr is None:
            return

        if self.local_lr_decay_coefficient is None:
            client.lr = self.local_lr
            return

        power_rank = self.global_epoch + group_index
        lr = self.local_lr * np.power(self.local_lr_decay_coefficient, power_rank)
        client.lr = lr

    def _aggregation(self):
        last_clients = [group[-1] for group in self.client_subtask_groups]
        weights = self.trainset_proportions(clients=last_clients)
        weights = torch.from_numpy(np.array(weights)).to(self.device)

        new_state_dict = gradient_based_fedavg(
            original_params=self.global_model.state_dict(),
            weights=weights,
            gradients=[client.gradient() for client in last_clients]
        )
        self.global_model.load_state_dict(state_dict=new_state_dict)
