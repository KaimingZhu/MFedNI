"""
@Encoding:      UTF-8
@File:          client.py

@Introduction:  definition of Client
@Author:        Kaiming Zhu
@Date:          2023/8/8 1:04
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from ..utils.path import is_path_exist, ensure_make_path
from ..utils.model import copy_state_dict, diff_between_state_dicts, sum_between_state_dicts


class Client:
    """ Base Class, Clients in Distributed Learning, providing common method and callback."""
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
        # dataset
        self.train_loader: DataLoader = train_loader
        self.test_loader: DataLoader = test_loader

        # model training
        self.model: nn.Module = model
        self.criterion: _Loss = criterion
        self.optimizer: Optimizer = optimizer
        self.scheduler: Optional[Any] = scheduler
        self.local_epoch: int = local_epoch
        self._device: str = device

        # archiving metrics
        self.archive_path: Optional[str] = None
        self.original_state_dict: Dict = copy_state_dict(self.model.state_dict())
        self._current_global_epoch = 0

    @property
    def lr(self) -> float:
        """learning rate of client model.
        See Also:
            https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no
        """
        return self.optimizer.param_groups[0]['lr']

    @lr.setter
    def lr(self, lr: float):
        """set learning rate for client model.
        Args:
            lr(float): new learning rate.
        See Also:
            https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no
        """
        self.optimizer.param_groups[0]['lr'] = lr

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str):
        self._device = device
        self.model.to(device)
        self.criterion.to(device)

    @property
    def current_global_epoch(self) -> int:
        return self._current_global_epoch

    @current_global_epoch.setter
    def current_global_epoch(self, current_global_epoch: int):
        old_value = self._current_global_epoch
        self._current_global_epoch = current_global_epoch
        self._current_global_epoch_did_set(old_value=old_value)

    def _current_global_epoch_did_set(self, old_value: int):
        """Event Invoking for subclass: current global epoch has changed."""
        pass

    def trainset_length(self) -> int:
        return len(self.train_loader.dataset)

    def testset_length(self) -> int:
        return len(self.test_loader.dataset)

    def state_dict(self) -> Dict:
        """ Method to make a copy of client model's state dict.
        Notes:
            Model's weights in pytorch is represented as a Dict object, naming `state_dict`.
            To make it more straight-forward, you can just regard it as `params_by_name`.

            KV(key-value) in `state_dict` is formatted by following rules:
                - key(str): name of weight, usually named as `layer`, `layer_bias`, `layer_weights`...
                - value(torch.Tensor): weights of corresponding key.

        See Also:
            - zh-CN: https://zhuanlan.zhihu.com/p/98563721
            - en-US: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
            - API:   torch.nn.Modules.state_dict()
        """
        return copy_state_dict(self.model.state_dict())

    def update_with_state_dict(self, state_dict: Dict):
        """method to set client's model w.r.t. `state_dict`.

        **Latex**
            \\\\theta_{new} = \\\\theta_{received}.

        Args:
            state_dict(Dict): target state dict.

        Notes:
            - state dict update in Federated Learning has skip for the param `num_batches_tracked`,
            that is because `num_batches_tracked` is a non-trainable LongTensor. And it is same
            for all clients for the given datasets.
            - See the implementation of FedAvg, in
            *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization* for more
            details.

        See Also:
            - API: Client.state_dict(self)
            - Code: [FedBN/federated/fed_office#L111](https://github.com/med-air/FedBN/blob/master/federated/fed_office.py#L111)
        """

        # keep `num_batches_tracked` always same to clients model.
        self_state_dict = self.model.state_dict()
        for key in state_dict.keys():
            if 'num_batches_tracked' in key and key in self_state_dict:
                param: torch.Tensor = state_dict[key]
                param.data.copy_(self_state_dict[key])

        self.model.load_state_dict(copy_state_dict(state_dict), strict=True)
        self.original_state_dict = copy_state_dict(state_dict)

    def gradient(self) -> Dict:
        """method to get clients gradient.
        Latex:
            gradient = \nabla \theta = \theta_{new} - theta_{original}

        Returns:
            a Dict object refers to clients gradient
        """
        return diff_between_state_dicts(self.model.state_dict(), self.original_state_dict)

    def update_with_gradient(self, gradient: Dict):
        """method to update client's model w.r.t. `gradient`.
        Latex:
            gradient = \nabla \theta = \theta_{new} - theta_{original}
            \theta^{new}_{client} = \theta_{client} + gradient

        Args:
            gradient(Dict): the gradient you want this client to update with. Which should
            comply the format of `nn.Module.state_dict()`.
        See Also:
            API: Client.state_dict(self)
            API: torch.nn.Modules.state_dict()
        """

        new_state_dict = sum_between_state_dicts(self.state_dict(), copy_state_dict(gradient))
        self.update_with_state_dict(new_state_dict)

    def eval(self, model: nn.Module = None, state_dict: Optional[Dict[str, Any]] = None, need_train: bool = True, need_test: bool = True) -> Dict:
        """eval model with data holding by client, return the result in dict.
        Args:
            model(nn.Module): model you need to eval. If this value is `None`,
            client will return the eval result of its own model. Default: `None`.

            state_dict(Optional, Dict): state dict you need to eval. If value is
            `None`, client will return the eval result with `model`'s state dict.

            need_train(bool): a boolean value indicating if you need metrics
            eval on trainset. If not, those key-value will not be calculated.
            You could accelerate the evaluation with it. Default: `True`.

            need_test(bool): a boolean value indicating if you need metrics
            eval on testset. If not, those key-value will not be calculated.
            You could accelerate the evaluation with it. Default: `True`.

        Returns:
            a Dict[str, float] contains four keys:
                - "train_acc": average accuracy on trainset, ranging 0 to 1.
                - "test_loss": average loss on testset.
                - "test_acc": average accuracy on testset, ranging 0 to 1.
                - "test_loss": average loss on testset.
        """
        eval_model: nn.Module = self.model
        if model is not None:
            eval_model = model
        eval_model.to(self.device)

        original_state_dict: Optional[Dict] = None
        if state_dict is not None:
            original_state_dict = copy_state_dict(eval_model.state_dict())
            eval_model.load_state_dict(copy_state_dict(state_dict=state_dict))

        destination: Dict[str: float] = {}
        if need_train:
            acc, loss = self._calculate_acc_loss_for(model=eval_model, loader=self.train_loader)
            destination["train_acc"] = acc
            destination["train_loss"] = loss

        if need_test:
            acc, loss = self._calculate_acc_loss_for(model=eval_model, loader=self.test_loader)
            destination["test_acc"] = acc
            destination["test_loss"] = loss

        if original_state_dict is not None:
            eval_model.load_state_dict(original_state_dict)
        return destination

    def _eval_original_model(self, need_train: bool = True, need_test: bool = True) -> Dict:
        """Calculate Evaluation metrics on Original Model, i.e., model with `original_state_dict`

        Args:
            need_train(bool): a boolean value indicating if you need metrics
            eval on trainset. If not, those key-value will not be calculated.
            You could accelerate the evaluation with it. Default: `True`.

            need_test(bool): a boolean value indicating if you need metrics
            eval on testset. If not, those key-value will not be calculated.
            You could accelerate the evaluation with it. Default: `True`.

        Returns:
            a Dict[str, float] object. It contains four keys:
            "train_acc": average accuracy on trainset, ranging [0,1];
            "test_loss": average loss on testset;
            "test_acc": average accuracy on testset, ranging [0,1];
            "test_loss": average loss on testset.

        See Also:
            Client.eval(model: nn.Module, need_train: bool = True, need_test: bool = True)
        """
        latest_state_dict = self.state_dict()
        self.model.load_state_dict(self.original_state_dict)

        destination = self.eval(need_train=need_train, need_test=need_test)
        self.model.load_state_dict(latest_state_dict)
        return destination

    def _calculate_acc_loss_for(self, model: nn.Module, loader: DataLoader) -> (float, float):
        if model.training:
            model.eval()

        with torch.no_grad():
            length = torch.zeros(1).to(self.device)
            sum_correct_amount = torch.zeros(1).to(self.device)
            sum_loss = torch.zeros(1).to(self.device)
            for i, (datas, labels) in enumerate(loader):
                datas = datas.to(self.device)
                labels = labels.to(self.device)
                outputs = model(datas)
                inferred_labels = torch.argmax(outputs, dim=-1)

                losses = self.criterion(outputs, labels)
                is_correct_list = (inferred_labels == labels)
                sum_correct_amount = sum_correct_amount + torch.sum(is_correct_list)
                sum_loss = sum_loss + torch.sum(losses)
                length = length + datas.size(0)

            avg_acc: float = (sum_correct_amount / length).detach().cpu().numpy().tolist()[0]
            avg_loss: float = (sum_loss / length).detach().cpu().numpy().tolist()[0]

            return avg_acc, avg_loss

    def train(self):
        self._local_training_will_start()
        self._train_model()
        self._local_training_will_finish()

    def _local_training_will_start(self):
        """Call back for subclasses: local training is going to start.
        Subclass can override this method, and perform necessary pre-process before training.
        """
        pass

    def _train_model(self):
        """ Function to describe the algorithm for training client's model.
        Notes:
            For most cases, this method do not need overriden.
        """
        for current_epoch in range(self.local_epoch):
            if not self.model.training:
                self.model.train()

            for i, (datas, labels) in enumerate(self.train_loader):
                datas = datas.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(datas)
                losses = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self._training_epoch_will_finish(epoch=current_epoch)

    def _training_epoch_will_finish(self, epoch: int):
        """Call back for listeners: model training will finish at epoch {epoch}.
        Listener can register this callback, record / archive necessary metrics.
        """
        pass

    def _local_training_will_finish(self):
        """Call back for subclasses: local training will finish.
        Subclass can override this method, and perform necessary post-process after training.
        """
        pass

    def archive_training_checkpoint(self, filename: Optional[str] = None, global_epoch: Optional[int] = None):
        """ archive training checkpoints, to support Client's training recovering to current status.
        Notes:
            Training checkpoints, means state_dicts for following variables. If you just need to save model,
            gradients or e.t.c, please use `Client._archive_model_related_info()`
            - self.model
            - self.optimizer
            - self.criterion
            - self.scheduler

        Args:
            global_epoch(Optional, int): Global epoch that need to archive for. Default is None.
            If default value has taken, it will take `self.current_global_epoch` instead.

            filename(Optional, str): Name of checkpoint file. This file is expected to be saved
            in the archiving folder for `global_epoch`. Default is None. If default value has taken,
            it will take `self._default_training_checkpoint_filename()` instead.

        See Also:
            Client._archive_model_related_info()
            Client._default_training_checkpoint_archive_folder()
            Client._default_training_checkpoint_filename()
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "criterion": self.criterion.state_dict(),
            "optim": self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if filename is None:
            filename = self._default_training_checkpoint_filename()
        if global_epoch is None:
            global_epoch = self.current_global_epoch
        self.archive_model_related_info(
            info_dict=checkpoint,
            filename=filename,
            global_epoch=global_epoch
        )

    def archive_model_related_info(self, info_dict: Dict, filename: str, global_epoch: Optional[int] = None):
        """ archive model related info to the archiving folder for `global_epoch`.
        Args:
            info_dict(Dict): Dict you need to archive.
            filename(str): Name of the file.
            global_epoch(Optional, int): Global epoch that need to archive for. Default is None.
            If default value has taken, it will take `self.current_global_epoch` instead.

        See Also:
            Client._default_training_checkpoint_archive_folder()
        """
        if global_epoch is None:
            global_epoch = self.current_global_epoch
        folder = self._default_training_checkpoint_archive_folder(global_epoch=global_epoch)

        ensure_make_path(folder)
        torch.save(info_dict, folder + filename)

    def recover_training_checkpoint(self, filename: Optional[str] = None, global_epoch: Optional[int] = None) -> bool:
        """try to recover on given epoch, return the result if it has recovered.
        Args:
            global_epoch(Optional, int): Global epoch that need to recover. Default is None.
            If default value has taken, it will take `self.current_global_epoch` instead.

            filename(Optional, str): Name of checkpoint file. This file is expected to be saved
            in the archiving folder for `global_epoch`. Default is None. If default value has taken,
            it will take `self._default_training_checkpoint_filename()` instead.

        Returns:
            a boolean indicating if it has recovered.

        Warnings:
            Exception will throw if path not exist.

        See Also:
            Client.has_archived_training_checkpoint(self, folder, filename)
            Client._default_training_checkpoint_archive_folder()
            Client._default_training_checkpoint_filename()
        """
        if filename is None:
            filename = self._default_training_checkpoint_filename()
        if global_epoch is None:
            global_epoch = self.current_global_epoch
        assert self.has_archived_training_checkpoint(global_epoch=global_epoch, filename=filename)

        folder = self._default_training_checkpoint_archive_folder(global_epoch=global_epoch)
        checkpoint = torch.load(folder + filename)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])
        self.criterion.load_state_dict(checkpoint["criterion"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        return True

    def has_archived_training_checkpoint(self, filename: Optional[str] = None, global_epoch: Optional[int] = None) -> bool:
        """A convenience function to judge if a checkpoint file has archived.
        Args:
            global_epoch(Optional, int): Global epoch that need to be judged. Default is None.
            If default value has taken, it will take `self.current_global_epoch` instead.

            filename(Optional, str): Name of checkpoint file. This file is expected to be saved
            in the archiving folder for `global_epoch`. Default is None. If default value has taken,
            it will take `self._default_training_checkpoint_filename()` instead.

        Returns:
            a boolean value, means if it has recovered.

        See Also:
            Client._default_training_checkpoint_archive_folder()
            Client._default_training_checkpoint_filename()
        """
        if filename is None:
            filename = self._default_training_checkpoint_filename()
        folder = self._default_training_checkpoint_archive_folder(global_epoch=global_epoch)
        return is_path_exist(folder + filename)

    def _default_training_checkpoint_archive_folder(self, global_epoch: Optional[int] = None) -> str:
        """ return the preferred path to archive training checkpoint """
        if global_epoch is None:
            global_epoch = self.current_global_epoch
        return self.archive_path + f"epoch_{global_epoch}//"

    def _default_training_checkpoint_filename(self) -> str:
        return "checkpoint.pt"

    def _metrics_archive_folder(self):
        return self.archive_path

    @staticmethod
    def _metrics_archive_filename():
        return "metrics.pt"

    def track_for_event(self, *events, need_model: bool = True, need_grad: bool = True, need_metrics: bool = True):
        if need_model:
            state_dict = self.state_dict()
            for event in events:
                self.archive_model_related_info(info_dict={"model": state_dict}, filename=f"{event}.pt")

        if need_grad:
            gradient = self.gradient()
            for event in events:
                self.archive_model_related_info(info_dict={"grad": gradient}, filename=f"{event}_grad.pt")

        if need_metrics:
            metrics_by_key = self.eval()
            metrics_dict = {event: metrics_by_key for event in events}
            self._archive_metrics(metrics=metrics_dict)

    def _archive_metrics(self, metrics: Any, global_epoch: Optional[int] = None):
        if global_epoch is None:
            global_epoch = self.current_global_epoch

        folder = self._metrics_archive_folder()
        filename = self._metrics_archive_filename()
        ensure_make_path(folder)

        metrics_by_global_epoch = {}
        if is_path_exist(folder + filename):
            metrics_by_global_epoch = torch.load(folder + filename)

        metrics_in_this_epoch = metrics
        if global_epoch in metrics_by_global_epoch.keys():
            archived_metrics = metrics_by_global_epoch[global_epoch]
            if isinstance(archived_metrics, Dict):
                archived_metrics.update(metrics)
                metrics_in_this_epoch = archived_metrics
            elif isinstance(archived_metrics, List):
                archived_metrics.extend(metrics)
                metrics_in_this_epoch = archived_metrics
        metrics_by_global_epoch[global_epoch] = metrics_in_this_epoch
        torch.save(metrics_by_global_epoch, folder + filename)

    def event_metrics_dict_by_global_epoch(self) -> Dict:
        folder = self._metrics_archive_folder()
        filename = self._metrics_archive_filename()
        if not is_path_exist(folder + filename):
            return {}
        else:
            return torch.load(folder + filename)

    def gradient_snapshot(self, event: str, global_epoch: Optional[int]) -> Optional[Dict]:
        if global_epoch is None:
            global_epoch = self.current_global_epoch

        folder = self._default_training_checkpoint_archive_folder(global_epoch=global_epoch)
        filename = f"{event}_grad.pt"
        if not (is_path_exist(folder) and is_path_exist(folder + filename)):
            return None

        info_dict = torch.load(folder + filename)
        key = "grad"
        if not key in info_dict:
            return None
        return info_dict[key]

    def model_snapshot(self, event: str, global_epoch: Optional[int]) -> Optional[Dict]:
        if global_epoch is None:
            global_epoch = self.current_global_epoch

        folder = self._default_training_checkpoint_archive_folder(global_epoch=global_epoch)
        filename = f"{event}.pt"
        if not (is_path_exist(folder) and is_path_exist(folder + filename)):
            return None

        info_dict = torch.load(folder + filename)
        key = "model"
        if not key in info_dict:
            return None
        return info_dict[key]

    def metrics_dict_snapshot(self, event: str, global_epoch: Optional[int]) -> Optional[Dict]:
        if global_epoch is None:
            global_epoch = self.current_global_epoch

        metrics_by_global_epoch = self.event_metrics_dict_by_global_epoch()
        if global_epoch not in metrics_by_global_epoch.keys():
            return None

        metrics_dict_by_event = metrics_by_global_epoch[global_epoch]
        if event not in metrics_dict_by_event.keys():
            return None

        return metrics_dict_by_event[event]
