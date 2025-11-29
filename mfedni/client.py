"""
@Encoding:      UTF-8
@File:          client.py

@Introduction:  Definition of mfedni.Client
@Author:        Kaiming Zhu, Songcan Yu, Feiyuan Liang.
@Date:          2023/12/27 21:52
"""

from typing import Any, Optional, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

import simulator.core as core


class Client(core.Client):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: _Loss,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        local_epoch: int = 5,
        device: str = "cpu",
        alpha: Optional[float] = None
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

        self.alpha = alpha
        self.global_prototypes_by_label: Dict[int, torch.Tensor] = {}
        self.prototype_by_label: Dict[int, torch.Tensor] = {}

        self._cos = nn.CosineSimilarity(dim=-1).to(self.device)
        self._prototype_loss = nn.CrossEntropyLoss()

    def _should_train_with_prototype(self) -> bool:
        return self.alpha is not None and len(self.global_prototypes_by_label.items()) != 0

    def _train_model(self):
        """ Function to describe the algorithm for training client's model."""
        for current_epoch in range(self.local_epoch):
            if not self.model.training:
                self.model.train()

            for data_by_key in self.train_loader:
                self.optimizer.zero_grad()
                datas = [data_by_key[key].to(self.device) for key in self.train_loader.dataset.data_keys]
                labels = data_by_key[self.train_loader.dataset.target_key].to(self.device)

                latent_representations, outputs = self.model(*datas, need_extract_feat=True)
                losses = self.criterion(outputs, labels)

                if self._should_train_with_prototype():
                    prototype_loss = self._calculate_prototype_loss(
                        latent_representations=latent_representations,
                        labels=labels
                    )
                    losses += self.alpha * prototype_loss

                losses.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self._training_epoch_will_finish(epoch=current_epoch)

    def _calculate_prototype_loss(self, latent_representations: torch.Tensor, labels: torch.Tensor):
        """prototype loss w.r.t. latent representations, used to train target model.
        Args:
            latent_representations(torch.Tensor): latent representation of datas, extracted from model.
            labels(torch.Tensor): labels of those latent representations.

        Returns:
            (torch.Tensor): loss w.r.t. latent representations you given.

        References:
            [1] https://discuss.pytorch.org/t/inplace-operation-breaks-computation-graph/91754/2
            [2] https://stackoverflow.com/questions/53987906/how-to-multiply-a-tensor-row-wise-by-a-vector-in-pytorch
        """
        # retrieve global prototypes
        global_prototypes = []
        for label in sorted(self.global_prototypes_by_label.keys()):
            global_prototypes.append(self.global_prototypes_by_label[label])
        global_prototypes = torch.stack(global_prototypes)

        # calculate similarities w.r.t. each global prototypes
        each_label_similarities = []
        for global_prototype in global_prototypes:
            target_prototype = torch.zeros_like(latent_representations)
            for index in range(0, target_prototype.size(0)):
                target_prototype[index] = global_prototype

            similarities = self._cos(latent_representations, target_prototype)
            each_label_similarities.append(similarities)

        stacked_similarities = torch.stack(each_label_similarities, dim=1)
        loss = self._prototype_loss(stacked_similarities, labels)
        return loss

    def _local_training_will_finish(self):
        # calculate latest_prototypes
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            prototypes_by_label: Dict[int, List[torch.Tensor]] = {}
            for data_by_key in self.train_loader:
                datas = [data_by_key[key].to(self.device) for key in self.train_loader.dataset.data_keys]
                labels = data_by_key[self.train_loader.dataset.target_key].to(self.device)
                latent_representations, _ = self.model(*datas, need_extract_feat=True)

                for latent_representation, label in zip(latent_representations, labels):
                    if label.item() not in prototypes_by_label.keys():
                        prototypes_by_label[label.item()] = []
                    prototypes_by_label[label.item()].append(latent_representation)

            for label, prototypes in prototypes_by_label.items():
                stacked_prototype = torch.stack(prototypes)
                self.prototype_by_label[label] = torch.mean(stacked_prototype, dim=0)

    def _calculate_acc_loss_for(self, model: nn.Module, loader: DataLoader) -> (float, float):
        if model.training:
            model.eval()

        with torch.no_grad():
            length = torch.zeros(1).to(self.device)
            sum_correct_amount = torch.zeros(1).to(self.device)
            sum_loss = torch.zeros(1).to(self.device)
            for data_by_key in loader:
                data_keys = sorted(list(data_by_key.keys()))
                data_keys = list(filter(lambda x: x != "label", data_keys))
                datas = [data_by_key[key].to(self.device) for key in data_keys]
                labels = data_by_key["label"].to(self.device)
                outputs = model(*datas)

                inferred_labels = torch.argmax(outputs, dim=-1)
                losses = self.criterion(outputs, labels)
                is_correct_list = (inferred_labels == labels)
                sum_correct_amount = sum_correct_amount + torch.sum(is_correct_list)
                sum_loss = sum_loss + torch.sum(losses)
                length = length + labels.size(0)

            avg_acc: float = (sum_correct_amount / length).detach().cpu().numpy().tolist()[0]
            avg_loss: float = (sum_loss / length).detach().cpu().numpy().tolist()[0]

            return avg_acc, avg_loss
