"""
@Encoding:      UTF-8
@File:          aggregator.py

@Introduction:  common aggregator for distributed learning simulation
@Author:        Kaiming Zhu
@Date:          2023/8/10 20:15
"""

from typing import Dict

import torch

from .model import copy_state_dict


def fedavg(param_names: [str], weights: torch.Tensor, state_dicts: [Dict]) -> Dict:
    """Federated Learning Average Aggregation(FedAvg).
    Math:
        \theta_{new} = \sum^{N}_{i=0} \alpha_{i} \theta_{i}.
        where: \alpha_{i} is the weight of client_{i}'s parameters, usually calculated by dataset ratio to overall.

    Args:
        param_names([str]): names of params.
        weights(torch.Tensor): weight of clients, weights[i] means the weight of client_{i}.
        state_dicts([Dict]): params from clients, state_dicts[i] means the param of client_{i}.

    See Also:
        - McMahan, H.B., Moore, E., Ramage, D., Hampson, S., & Arcas, B.A. (2016).
          Communication-Efficient Learning of Deep Networks from Decentralized Data.
          International Conference on Artificial Intelligence and Statistics.
        - https://github.com/WHDY/FedAvg/blob/master/use_pytorch

    Warnings:
        Please ensure that weight.size(0) == len(state_dict), or it will raise an error.

    Returns:
        a dict, compling to the format of `torch.nn.Module.state_dict()`, indicating the result of aggregation.

    Examples:
        >>> param_a = {"a": torch.Tensor([1, 2, 3]), "b": torch.Tensor([2, 3, 4])}
        >>> param_b = {"a": torch.Tensor([2, 3, 4]), "b": torch.Tensor([3, 4, 5])}
        >>> weights = torch.Tensor([0.2, 0.8])
        >>> keys = ["a", "b"]
        >>> print(fedavg(keys, weights, [param_a, param_b]))
        {'a': tensor([1.8000, 2.8000, 3.8000]), 'b': tensor([2.8000, 3.8000, 4.8000])}
        >>> keys = ["a"]
        >>> print(fedavg(keys, weights, [param_a, param_b]))
        {'a': tensor([1.8000, 2.8000, 3.8000])}
        >>> keys = ["a", "b"]
        >>> weights = torch.Tensor([0.5, 0.5])
        >>> print(fedavg(keys, weights, [param_a, param_b]))
        {'a': tensor([1.5000, 2.5000, 3.5000]), 'b': tensor([2.5000, 3.5000, 4.5000])}
    """
    assert len(state_dicts) > 0
    assert weights.size(0) == len(state_dicts), "length should be equal between weights and state_dicts"

    destination: Dict[str, torch.Tensor] = {param_name: None for param_name in param_names}
    for param_name in destination.keys():
        params = [state_dict[param_name] for state_dict in state_dicts]
        weighted_params = [param * weight for (param, weight) in zip(params, weights)]
        destination[param_name] = torch.sum(torch.stack(weighted_params, dim=0), dim=0)

    return destination


def gradient_based_fedavg(original_params: [Dict], weights: torch.Tensor, gradients: [Dict]) -> [Dict]:
    """Federated Learning Average Aggregation(FedAvg).
    Math:
        \theta^{*}_{global} = \theta_{global} + \sum^{N}_{i=0} \alpha_{i} \nabla theta_{i}.
        \nabla_theta_{i} = \theta^{*}_{i} - \theta_{i}

        where:
            - \alpha_{i} is the weight of client_{i}'s parameters, usually calculated by dataset ratio to overall.
            - \nabla theta_{i} is the gradient of client_{i}.
            - \theta_{i} is the params of client_{i}, before local training.
            - \theta^{*}_{i} is the params of client_{i}, after local training.

    Args:
        original_params(Dict): param from global model.
        weights(torch.Tensor): weight of clients, weights[i] means the weight of client_{i}.
        gradients([Dict]): gradients from clients, gradients[i] means the gradient of client_{i}.

    Warnings:
        Please ensure that weight.size(0) == len(gradients), or it will raise an error.

    Returns:
        a dict, compling to the format of `torch.nn.Module.state_dict()`, indicating the result of aggregation.

    Examples:
        >>> gradient_a = {"a": torch.Tensor([1, 2, 3]), "b": torch.Tensor([2, 3, 4])}
        >>> gradient_b = {"a": torch.Tensor([2, 3, 4]), "b": torch.Tensor([3, 4, 5])}
        >>> original_params = {"a": torch.Tensor([2, 3, 4])}
        >>> weights = torch.Tensor([0.2, 0.8])
        >>> print(gradient_based_fedavg(original_params, weights, [gradient_a, gradient_b]))
        {'a': tensor([3.8000, 5.8000, 7.8000])}
        >>> original_params = {"a": torch.Tensor([2, 3, 4]), "b": torch.Tensor([1, 1, 1])}
        >>> print(gradient_based_fedavg(original_params, weights, [gradient_a, gradient_b]))
        {'a': tensor([3.8000, 5.8000, 7.8000]), 'b': tensor([3.8000, 4.8000, 5.8000])}
        >>> weights = torch.Tensor([0.5, 0.5])
        >>> print(gradient_based_fedavg(original_params, weights, [gradient_a, gradient_b]))
        {'a': tensor([3.5000, 5.5000, 7.5000]), 'b': tensor([3.5000, 4.5000, 5.5000])}
    """

    assert len(gradients) > 0
    assert weights.size(0) == len(gradients), "length should be equal between weights and state_dicts"

    destination: Dict[str, torch.Tensor] = copy_state_dict(original_params)
    for param_name in destination.keys():
        param_gradients = [gradient[param_name] for gradient in gradients]
        weighted_gradients = [param_gradient * weight for (param_gradient, weight) in zip(param_gradients, weights)]
        destination[param_name] = destination[param_name] + torch.sum(torch.stack(weighted_gradients, dim=0), dim=0)

    return destination