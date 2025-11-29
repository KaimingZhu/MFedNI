"""
@Encoding:      UTF-8
@File:          torch_model.py

@Introduction:  interface to get a model implemented in pytorch, factory pattern.
@Author:        Kaiming Zhu
@Date:          2023/8/6 19:45
"""

from enum import Enum
from typing import Tuple, Optional, Callable, Dict, Any

from torch import nn

from .initializer import Initializer
from .resnet import ResNet
from .lenet import LeNet5

from .modal_merge_model import ModalMergeModel
from .modal_concat_model import ModalConcatModel
from .cmtff import CMTFFModel, AblationMLPModel, AblationTransformerModel


class ModelFactory(Enum):
    # Single Modal
    LeNet5 = 1,
    ResNet20 = 2,
    ResNet32 = 3,
    ResNet44 = 4,
    ResNet56 = 5
    ResNet110 = 6,
    ResNet1202 = 7,

    # Multiple Modal
    ModalMergeModel = 8,
    ModalConcatModel = 9,
    CMTFFModel = 10,
    AblationMLPModel = 11,
    AblationTransformerModel = 12

    def instance(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        initializer: Optional[Initializer] = Initializer.KaimingNormal
    ) -> nn.Module:

        instance_method_by_enum: Dict[ModelFactory: Callable[[Any, Tuple[int], Tuple[int]], nn.Module]] = {
            ModelFactory.LeNet5: LeNet5.instance,
            ModelFactory.ResNet20: ResNet.resnet20_instance,
            ModelFactory.ResNet32: ResNet.resnet32_instance,
            ModelFactory.ResNet44: ResNet.resnet44_instance,
            ModelFactory.ResNet56: ResNet.resnet56_instance,
            ModelFactory.ResNet110: ResNet.resnet110_instance,
            ModelFactory.ResNet1202: ResNet.resnet1202_instance,
        }

        model: Optional[nn.Module] = None
        instance_method = instance_method_by_enum[self]
        if instance_method is not None:
            model = instance_method(input_shape, output_shape)

        if model is None:
            raise AssertionError
        if initializer is not None:
            initializer.apply(model)

        return model

    def multi_modal_instance(
        self,
        *input_shapes: Tuple[Tuple[int]],
        output_shape: Tuple[int],
        initializer: Optional[Initializer] = Initializer.KaimingNormal
    ) -> nn.Module:

        instance_method_by_enum: Dict[ModelFactory: Callable] = {
            ModelFactory.ModalMergeModel: ModalMergeModel.instance,
            ModelFactory.ModalConcatModel: ModalConcatModel.instance,
            ModelFactory.CMTFFModel: CMTFFModel.instance,
            ModelFactory.AblationMLPModel: AblationMLPModel.instance,
            ModelFactory.AblationTransformerModel: AblationTransformerModel.instance,
        }

        model: Optional[nn.Module] = None
        instance_method = instance_method_by_enum[self]
        if instance_method is not None:
            input_shapes = list(input_shapes)
            model = instance_method(*input_shapes, output_shape=output_shape)

        if model is None:
            raise AssertionError
        if initializer is not None:
            initializer.apply(model)

        return model
