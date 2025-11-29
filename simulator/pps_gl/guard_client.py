"""
@Encoding:      UTF-8
@File:          client.py

@Introduction:  Definition of pps_gl.GuardClient
@Author:        Kaiming Zhu
@Date:          2023/8/17 22:23
"""

from typing import Any, Optional

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

import simulator.core as core


class GuardClient(core.Client):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: _Loss,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        local_epoch: int = 20,
        device: str = "cpu"
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            local_epoch=local_epoch,
            device=device
        )

    def _current_global_epoch_did_set(self, old_value: int):
        self.track_for_event("initial", need_grad=False)

    def _parameter_boardcast_did_finish(self):
        self.track_for_event("boardcast", need_grad=False)

    def _local_training_will_start(self):
        self.track_for_event("begin", need_grad=False)

    def _training_epoch_will_finish(self, epoch: int):
        metric_by_key = self.eval(need_train=False)
        test_acc, test_loss = metric_by_key["test_acc"], metric_by_key["test_loss"]
        print(f"epoch {epoch}: test acc = {test_acc}, test loss = {test_loss}")

    def _local_training_will_finish(self):
        self.track_for_event("train", "send")
