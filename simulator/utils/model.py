"""
@Encoding:      UTF-8
@File:          model.py

@Introduction:  utils for pytorch model
@Author:        Kaiming Zhu
@Date:          2023/8/9 4:48
"""

from typing import Dict

import torch


def copy_state_dict(state_dict: Dict) -> Dict:
    """ Method to make a `clean` copy of another state dict from any {nn.Modules}.

    This method can make a clean copy of model's state dict, which means: \\
        - the copy one has same key-value as original.
        - tensors in the copy one has detached from the original computation graph, i.e.,
          further propagations on the copy one will not affect the original.

    Args:
        state_dict(Dict[str, torch.Tensor]): state dict you need to copy.

    Returns:
        the cloned and detached state-dict.

    See Also:
        https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
        https://pytorch.org/docs/stable/tensors.html#torch.Tensor.new_tensor
    """
    destination: Dict[str, torch.Tensor] = {}
    for (param_name, param) in state_dict.items():
        destination[param_name] = param.clone().detach()
    return destination


def diff_between_state_dicts(minuend: Dict, subtrahend: Dict) -> Dict:
    """Convenience method to calculate sum of state dicts, like `a - b`.

    **Latex**
        \\\\nabla \\\\theta_{return} = \\\\theta_{minuend} - \\\\theta_{subtrahend}

    Examples:
        >>> import torch
        >>> a = {'param_name': torch.Tensor([1, 2, 3])}
        >>> b = {'param_name': torch.Tensor([1, 1, 1])}
        >>> diff_between_state_dicts(a, b)
        {'param_name': tensor([0., 1., 2.])}

        >>> c = {'param_name': torch.Tensor([1, 2, 4])}
        >>> diff_between_state_dicts(a, c)
        {'param_name': tensor([0., 0., -1.])}

    Args:
        minuend(Dict): minuend of the diff.
        subtrahend(Dict): subtrahend of the diff.

    Returns:
        The diff between minuend and subtrahend, can regard as gradient.

    Warnings:
        - You should guarantee that `len(state_dicts) >= 0`, or it will raise an assertion error.
        - You should guarantee that all those state_dict having same key, and all those corresponding
          torch.Tensor have same shape.

    See Also:
        - CN version: https://zhuanlan.zhihu.com/p/98563721
        - EN version: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
        - API:        torch.nn.Modules.state_dict()

    **WIP**
        cc @kaiming, make a state dict wrapper, to enable operations like `c = a - b`, "d = a + b".
    """
    minuend = copy_state_dict(minuend)
    subtrahend = copy_state_dict(subtrahend)

    destination: Dict[str, torch.Tensor] = {}
    for param_name in minuend.keys():
        minuend_param = minuend[param_name]
        subtrahend_param = subtrahend[param_name]
        destination[param_name] = minuend_param - subtrahend_param
    return destination


def sum_between_state_dicts(*state_dicts) -> Dict:
    """Convenience method to calculate sum of state dicts, like `a + b + c`.

    **Latex**
        \\\\theta_{return} = \\\\sum^{N}_{i=0}\\\\theta_{i}.

    Examples:
        >>> import torch
        >>> a = {'param_name': torch.Tensor([1, 2, 3])}
        >>> b = {'param_name': torch.Tensor([1, 1, 1])}
        >>> sum_between_state_dicts(a, b)
        {'param_name': tensor([2., 3., 4.])}

        >>> c = {'param_name': torch.Tensor([1, 2, 4])}
        >>> sum_between_state_dicts(a, b, c)
        {'param_name': tensor([3., 5., 8.])}

    Args:
        state_dicts: a tuple, where all elem should be an instance of `state_dict` from {nn.Modules}.

    Returns:
        The sum between those state dicts.

    Warnings:
        - You should guarantee that `len(state_dicts) >= 0`, or it will raise an assertion error.
        - You should guarantee that all those state_dict having same key, and all those corresponding
          torch.Tensor have same shape.

    See Also:
        - CN version: https://zhuanlan.zhihu.com/p/98563721
        - EN version: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
        - API:        torch.nn.Modules.state_dict()

    **WIP**
        cc @kaiming, make a state dict wrapper, to enable operations like `c = a - b`, `d = a + b`.
    """
    assert len(state_dicts) > 0

    state_dicts = [copy_state_dict(state_dict) for state_dict in state_dicts]
    first_state_dict = state_dicts[0]
    if len(state_dicts) == 1:
        return first_state_dict

    destination: Dict[str, torch.Tensor] = {}
    for param_name in first_state_dict.keys():
        params = [state_dict[param_name] for state_dict in state_dicts]
        destination[param_name] = torch.sum(torch.stack(params), dim=0)
    return destination


def weighted_state_dict(state_dict: Dict, weight: float) -> Dict:
    """Convenience method to multiple state dict with weight, like `a * 0.1`.

    **Latex**
        \\\\theta_{return} = \\\\theta_{original} * weight

    Examples:
        >>> import torch
        >>> a = {'param_name': torch.Tensor([1, 2, 3])}
        >>> b = {'param_name': torch.Tensor([1, 1, 1])}
        >>> weighted_state_dict(a, weight=0.5)
        {'param_name': tensor([0.5000, 1.0000, 1.5000])}

        >>> b = {'param_name': torch.Tensor([1, 1, 1])}
        >>> weighted_state_dict(b, weight=0.8)
        {'param_name': tensor([0.8000, 0.8000, 0.8000])}

    Args:
        state_dict(Dict): an instance of `state_dict` from {nn.Modules}.
        weight(float): weight you given.

    Returns:
        Weighted state dict with respect to (w.r.t.) original.

    See Also:
        - CN version: https://zhuanlan.zhihu.com/p/98563721
        - EN version: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
        - API:        torch.nn.Modules.state_dict()
    """

    state_dict = copy_state_dict(state_dict)
    destination: Dict[str, torch.Tensor] = {}
    for param_key in state_dict.keys():
        param = state_dict[param_key]
        destination[param_key] = param * weight

    return destination


def is_state_dicts_equal(*state_dicts) -> bool:
    """Convenience method to figure out if state dicts equal, like `a == b == c`.

    **Latex**
        is_equal = \\\\theta_{0} == \\\\theta_{1} ... === \\\\theta_{n}.

    Examples:
        >>> import torch
        >>> a = {'param_name': torch.Tensor([1, 2, 3])}
        >>> b = {'param_name': torch.Tensor([1, 2, 3])}
        >>> is_state_dicts_equal(a, b)
        True

        >>> c = {'param_name': torch.Tensor([1, 2, 4])}
        >>> is_state_dicts_equal(a, c)
        False
        >>> is_state_dicts_equal(a, b, c)
        False

        >>> d = {'param_name': torch.Tensor([1, 2, 3, 4])}
        >>> is_state_dicts_equal(a, d)
        False

    Args:
        state_dicts: a tuple, where all elem should be an instance of `state_dict` from {nn.Modules}.

    Returns:
        a boolean indicate if these state_dicts equal.

    See Also:
        - CN version: https://zhuanlan.zhihu.com/p/98563721
        - EN version: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
        - API:        torch.nn.Modules.state_dict()

    **WIP**
        cc @kaiming, make a state dict wrapper, to enable operations like `c = a - b`, "d = a + b".
    """
    if len(state_dicts) <= 1:
        return True

    key_sets: [set] = [set(list(state_dict.keys())) for state_dict in state_dicts]
    is_all_length_equal = all([len(key_set) == len(key_sets[0]) for key_set in key_sets])
    if not is_all_length_equal:
        return False

    set_differences = [key_set.difference(key_sets[0]) for key_set in key_sets]
    is_all_have_same_keys = all([len(difference) == 0 for difference in set_differences])
    if not is_all_have_same_keys:
        return False

    for key in state_dicts[0].keys():
        params: [torch.Tensor] = [state_dict[key] for state_dict in state_dicts]

        param_shapes: [tuple] = [param.shape for param in params]
        is_all_shape_equal = all([shape == param_shapes[0] for shape in param_shapes])
        if not is_all_shape_equal:
            return False

        is_params_equal = all([(param == params[0]).all().cpu().numpy().tolist() for param in params])
        if not is_params_equal:
            return False

    return True