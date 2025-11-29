"""
@Encoding:      UTF-8
@File:          simulator.py

@Introduction:  Definition of gl.Simulator
@Author:        Kaiming Zhu
@Date:          2023/8/14 23:02
"""

from typing import Optional, Dict, List

import numpy as np
import torch
from torch import nn

from ..core import DLSimulator, Client
import simulator.gl as gl
from ..utils.aggregator import fedavg


class Simulator(DLSimulator):
    """Gossip Learning Simulator.
    See Also:
        Edge-assisted Gossiping Learning: Leveraging V2V Communications between Connected Vehicles.
    """

    def __init__(
        self,
        clients: [Client],
        global_model: nn.Module,
        global_epochs: int,
        archive_folder_name: Optional[str] = None,
        device: str = "cpu",
        clients_select_ratio: float = 1.0,
        subtask_divide_ratio: float = 0.2,
        local_lr: Optional[float] = 0.005,
        local_lr_decay_coefficient: Optional[float] = 0.999
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

    def _clients_did_change(self, old_value: [Client]):
        self._client_id_by_client = {client: i for i, client in enumerate(self._clients)}

    def _global_epoch_will_start(self, epoch: int):
        # Algorithm: Divide clients into several groups
        selected_client_index: [int] = self._generate_random_selected_index(ratio=self.clients_select_ratio)
        client_index_groups: [[int]] = self._make_groupped_indexs(
            indexs=selected_client_index,
            grouping_ratio=self.subtask_divide_ratio
        )
        self.client_subtask_groups = []
        for group in client_index_groups:
            if len(group) == 0:
                continue
            clients_group: [Client] = [self.clients[index] for index in group]
            self.client_subtask_groups.append(clients_group)

        # Visualize and track
        self._visualize_metrics_with_tracking(client_index_groups=client_index_groups)

    def _generate_random_selected_index(self, ratio: float) -> [int]:
        if ratio <= 0:
            return []

        indexs = np.arange(0, len(self.clients))
        np.random.shuffle(indexs)
        if ratio >= 1.0:
            return indexs
        else:
            select_slice = slice(0, int(ratio * len(self.clients)))
            return indexs[select_slice]

    def _make_groupped_indexs(self, indexs: [int], grouping_ratio: float) -> [[int]]:
        if grouping_ratio < 0 or grouping_ratio > 1:
            return [[] for _ in range(0, len(indexs))]
        elif grouping_ratio > 1:
            return [[indexs]]

        grouping_start_index = (np.arange(0, 1.0, grouping_ratio) * len(indexs)).astype('int64')
        index_groups: [[Client]] = []
        for (i, index) in enumerate(grouping_start_index):
            if i != len(grouping_start_index) - 1:
                start_index = index
                end_index = grouping_start_index[i + 1]
                index_groups.append(indexs[start_index:end_index].tolist())
            else:
                index_groups.append(indexs[index:].tolist())

        return index_groups

    def _boardcast_parameters(self):
        for group in self.client_subtask_groups:
            first_client: Client = group[0]
            first_client.update_with_state_dict(self.global_model.state_dict())

        # Event Invoking
        for client in self.clients:
            if not isinstance(client, gl.Client):
                continue
            client._parameter_boardcast_did_finish()

    def _local_training(self):
        for group_index, group in enumerate(self.client_subtask_groups):
            # Visualization
            print(f"Group {group_index} begin training.")
            for (i, client) in enumerate(group):
                # Visualization
                print(f"client {self._client_id_by_client[client]} begin training")

                # Algorithm
                self.update_client_lr_if_needed(client=client, group_index=i)
                if i == 0:
                    client.train()
                else:
                    last_client: Client = group[i-1]
                    client.update_with_state_dict(last_client.state_dict())
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

        new_state_dict = fedavg(
            param_names=list(self.global_model.state_dict().keys()),
            weights=weights,
            state_dicts=[client.state_dict() for client in last_clients]
        )
        self.global_model.load_state_dict(state_dict=new_state_dict)

    def _simulation_will_finish(self):
        self._visualize_metrics_with_tracking()

    def _visualize_metrics_with_tracking(self, client_index_groups: Optional[List[List[int]]] = None):
        # Visualize
        metrics_by_key = self.eval()
        test_acc, test_loss = metrics_by_key["test_acc"], metrics_by_key["test_loss"]
        print(f"global Epoch {self.current_epoch}: test acc {test_acc}, test loss {test_loss}")
        if client_index_groups is not None:
            print("clients group results:")
            for i, group in enumerate(client_index_groups):
                print(f"group {i}: {group}")

        # Track
        if client_index_groups is not None:
            metrics_by_key.update({"subtask_groups": client_index_groups})
        self.archive_metrics(
            epoch=self.current_epoch,
            metrics_dict=metrics_by_key
        )
