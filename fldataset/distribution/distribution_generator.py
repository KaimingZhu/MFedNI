"""
@Encoding:      UTF-8
@Filename:      distribution_generator.py

@Introduction:  The Distribution Generator
@Author:        Kaiming Zhu
@Date:          2023/07/12 19:27
@Reference:     https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseDistributionGenerator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, client_num: int) -> np.ndarray:
        pass

    def _post_processing_for_result(self, proportions: np.ndarray) -> np.ndarray:
        """post-processing for result calculated by subclasses.

            sometimes the sum of proportions may not equal to 1.0.
            That is mainly due to the precision error w.r.t. floating points.

            To ensure that sum(proportions) **ALWAYS** equal to 1.0,
            we will change the last elem to 1.0 - sum(other_elem), that is:
                proportion[-1] = 1 - (proportion[0] + proportion[1] + ... + proportion[-2])

            Examples:
                post_processing_for_result([0.1, 0.2, 0.3, 0.4])
                [0.1, 0.2, 0.3, 0.4]
                post_processing_for_result([0.1, 0.2, 0.3, 0.35])
                [0.1, 0.2, 0.3, 0.4]
                post_processing_for_result([0.1, 0.1, 0.1, 0.1])
                [0.1, 0.1, 0.1, 0.7]

            See Also:
                https://docs.python.org/3/tutorial/floatingpoint.html
        """
        proportions[-1] = 1.0 - np.sum(proportions[0:-1])
        return proportions


class UniformDistributed(BaseDistributionGenerator):
    """
    proportion between each user will be the same.
    """
    def __init__(self):
        pass

    def __call__(self, client_num: int) -> np.ndarray:
        proportions = np.repeat(1, client_num) / float(client_num)

        # sometimes the sum of proportions may not equal to 1.0, due to the floating point precision error.
        return super()._post_processing_for_result(proportions)


class LogNormalDistributed(BaseDistributionGenerator):
    """LogNormal Distributions, it means log(variable) complies to Gaussian Distribution.
        For any distribution ln(X) ~ N(mean, sigma*sigma), the mathematical expectation and variance equals to:
            - expectation: E(X) = exp(mean + sigma*sigma / 2.0)
            - variance: var(X) = (exp(mean + sigma*sigma) - 1) * exp(2 * mean + sigma*sigma)

        See Also:
            https://zh.wikipedia.org/zh-hans/%E5%AF%B9%E6%95%B0%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83
    """
    def __init__(self, mean: float = 0.0, sigma: float = 0.4):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, client_num: int) -> np.ndarray:
        distributions = np.random.default_rng().lognormal(mean=self.mean, sigma=self.sigma, size=client_num)
        proportions = distributions / np.sum(distributions)
        return super()._post_processing_for_result(proportions)


class DirichletDistributed(BaseDistributionGenerator):
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def __call__(self, client_num: int) -> np.ndarray:
        """Pass the number of client, and calculate the proportion distribution for each with respect to Dirichlet.

            Args:
                client_num (int): Number of clients for partition.

            Returns:
                numpy.ndarray: A numpy array, the value of index `i` represents the proportion of i-th client.

            See also:
                https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.dirichlet.html
                https://www.cnblogs.com/orion-orion/p/15897853.html
                https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py#L89

            Introduced By:
                Bayesian nonparametric federated learning of neural networks. In International Conference on Machine Learning (pp. 7252-7261). PMLR.
                Federated learning with matched averaging. arXiv preprint arXiv:2002.06440.
        """
        alpha_array = np.repeat(self.alpha, client_num)
        proportions = np.random.default_rng().dirichlet(alpha_array)
        return super()._post_processing_for_result(proportions)
