"""
@Encoding:      UTF-8
@File:          client.py

@Introduction:  Definition of pps_gl.Client
@Author:        Kaiming Zhu
@Date:          2023/8/17 22:23
"""

from typing import Any, Optional, Dict

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

import simulator.core as core
from ..utils.model import copy_state_dict, sum_between_state_dicts, weighted_state_dict


class Client(core.Client):
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

        self._received_gradient: Optional[Dict] = None

    def _current_global_epoch_did_set(self, old_value: int):
        self._received_gradient = None
        self.track_for_event("initial", need_grad=False)

    def _parameter_boardcast_did_finish(self):
        self.track_for_event("boardcast", need_grad=False)

    def update_with_gradient(self, gradient: Dict):
        self._received_gradient = copy_state_dict(gradient)
        self.archive_model_related_info(info_dict={"grad": self.state_dict()}, filename="extra_grad.pt")

        weighted_gradient = weighted_state_dict(state_dict=self._received_gradient, weight=self.lr)
        super().update_with_gradient(gradient=weighted_gradient)

    def _local_training_will_start(self):
        self.track_for_event("begin")

    def _training_epoch_will_finish(self, epoch: int):
        metric_by_key = self.eval(need_train=False)
        test_acc, test_loss = metric_by_key["test_acc"], metric_by_key["test_loss"]
        print(f"epoch {epoch}: test acc = {test_acc}, test loss = {test_loss}")

    def _local_training_will_finish(self):
        self.track_for_event("train")

        gradient = self.gradient()
        # do your extra post-process work on `gradient`, like replace gradients for privacy-preserving.
        pps_state_dict = sum_between_state_dicts(copy_state_dict(self.original_state_dict), gradient)
        self.model.load_state_dict(state_dict=pps_state_dict)

        self.track_for_event("send")
