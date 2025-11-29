"""
@Encoding:      UTF-8
@File:          distribution.py

@Introduction:  an enum for user to choose popular distribution
@Author:        Kaiming Zhu
@Date:          2023/7/19 20:03
@Reference:     https://github.com/SMILELab-FL/FedLab#datasets--data-partition
"""

from enum import Enum

from .distribution.sampler import IIDSampler, NonIIDSampler
from .distribution.distribution_generator import UniformDistributed, LogNormalDistributed, DirichletDistributed


class PopularDistribution(Enum):
    # https://github.com/SMILELab-FL/FedLab#1-balanced-iid-partition
    BalancedIID = 1
    # https://github.com/SMILELab-FL/FedLab#2-unbalanced-iid-partition
    UnbalancedIID = 2
    # https://github.com/SMILELab-FL/FedLab#3-hetero-dirichlet-partition
    # Introduced: Bayesian nonparametric federated learning of neural networks. ICML 2019.
    Dirichlet = 3

    def sampler(self):
        sampler_cls_by_enum = {
            PopularDistribution.BalancedIID: IIDSampler,
            PopularDistribution.UnbalancedIID: NonIIDSampler,
            PopularDistribution.Dirichlet: NonIIDSampler
        }

        sampler_cls = sampler_cls_by_enum[self]
        assert sampler_cls is not None, f"{self} do not register any sampler."

        sampler = sampler_cls()
        sampler.distribution_generator = self._distribution_generator()
        return sampler

    def _distribution_generator(self):
        generator_cls_by_enum = {
            PopularDistribution.BalancedIID: UniformDistributed,
            PopularDistribution.UnbalancedIID: LogNormalDistributed,
            PopularDistribution.Dirichlet: DirichletDistributed
        }

        generator_cls = generator_cls_by_enum[self]
        assert generator_cls is not None, f"{self} do not register any distribution generator."
        return generator_cls()
