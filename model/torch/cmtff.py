"""
@Encoding:      UTF-8
@File:          cmtff.py.py

@Introduction:  Definition of Cross-Modal Temporal Feature Fusion(CMTFF) modal
@Author:        Kaiming Zhu, Feiyuan Liang, Songcan Yu.
@Date:          2023/12/27 21:57
"""

from functools import reduce
import math
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Definition of PositionalEncoder
    Reference:
        https://pytorch.org/docs/1.6.0/generated/torch.nn.TransformerEncoder.html
    """
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        """Initializer of PositionalEncoder
        Args:
            embedding_size(int): embedding size
            dropout(float): hyper param for 'nn.Dropout', default is 0.1.
            max_len(int): default is 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeatureExtractor(nn.Module):
    """Definition of Self-Attention Based FeatureExtractor
    Reference:
        https://pytorch.org/docs/1.6.0/generated/torch.nn.TransformerEncoder.html
    """
    def __init__(self, input_shape: Tuple[int], embedding_size=128, nhead=4, num_layers=3):
        """Initializer of Self-Attention Based FeatureExtractor.
        Args:
            input_shape(Tuple[int]): shape of input, should be aranged as (N, time_slice, feature_dimension)
            embedding_size(int): size of embedding.
            nhead(int): param for `nn.TransformerEncoderLayer`
            num_layers(int): param for `nn.TransformerEncoder`
        """
        super().__init__()
        self.conv_layer = nn.Conv1d(in_channels=input_shape[-1], out_channels=embedding_size, kernel_size=3, stride=1, padding=1)
        self.positional_encoder = PositionalEncoder(embedding_size=embedding_size)

        encoder = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size * input_shape[-2], out_features=embedding_size, bias=True),
            nn.Sigmoid()
        )

        # Readonly property for sub-module
        self._embedding_size = embedding_size

    @property
    def embedding_size(self):
        return self._embedding_size

    def forward(self, x):
        output = x.permute(0, 2, 1)
        output = self.conv_layer(output)
        output = output.permute(0, 2, 1)

        output = self.positional_encoder(output)
        output = self.transformer_encoder(output)

        output = self.flatten(output)
        output = self.mlp(output)
        return output


class ModalFusionLayer(nn.Module):
    """Cross-modal attention based fusion module"""
    def __init__(self, modal_amount: int, embedding_size=128):
        super().__init__()
        self.weight_map = nn.Conv1d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=3, stride=1, padding=1)
        self.fusion_conv = nn.Conv1d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=modal_amount, stride=1, padding=0)

    def forward(self, *modal_datas):
        # calculate weight map of each modal
        modal_datas = list(modal_datas)
        stacked_data = torch.stack(modal_datas, dim=1)
        stacked_data = stacked_data.permute(0, 2, 1)
        weight_mapped_data = self.weight_map(stacked_data)
        weight_mapped_datas = []
        for i in range(0, weight_mapped_data.size()[-1]):
            weight_mapped_datas.append(weight_mapped_data[:, :, i])

        # modal-wise multiplication
        multiplied_datas = []
        for original_data, weight_map in zip(modal_datas, weight_mapped_datas):
            multiplied_datas.append(torch.mul(original_data, weight_map))

        # calculate latent representation
        latent_representation = torch.stack(multiplied_datas, dim=1)
        latent_representation = latent_representation.permute(0, 2, 1)
        latent_representation = self.fusion_conv(latent_representation)
        latent_representation = latent_representation.permute(0, 2, 1)
        return latent_representation


class CMTFFModel(nn.Module):
    """ Cross-Modal Temporal Feature Fusion(CMTFF) model for mfedni
    Notes:
        We defined `extractors` with `__set_attr__` and `__get_attr__`, rather than define and save them into list.
        That is because, parameters in list-saved layers will fail to move to GPU when we use `model.cuda()`. Please
        refer to the linkage below for more details.

    See Also:
        https://discuss.pytorch.org/t/some-tensors-getting-left-on-cpu-despite-calling-model-to-cuda/112915/7
    """
    embedding_size = 128

    def __init__(self, *input_shapes, output_shape):
        super().__init__()
        for index, input_shape in enumerate(input_shapes):
            extractor = FeatureExtractor(input_shape=input_shape, embedding_size=self.embedding_size)
            self.__setattr__(f"extractor{index}", extractor)
        self.fusion_module = ModalFusionLayer(modal_amount=len(input_shapes), embedding_size=self.embedding_size)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=output_shape[-1])
        )

    def forward(self, *modal_datas, need_extract_feat=False):
        extracted_datas: [torch.Tensor] = []
        for index, modal_data in enumerate(modal_datas):
            extractor: nn.Module = self.__getattr__(f"extractor{index}")
            extracted_datas.append(extractor(modal_data))

        merged_feat = self.fusion_module(*extracted_datas)
        merged_feat = torch.squeeze(merged_feat, dim=1)
        output = self.classifier(merged_feat)

        if need_extract_feat:
            return merged_feat, output
        else:
            return output

    @classmethod
    def instance(cls, *input_shapes: Tuple[Tuple[int]], output_shape: Tuple[int]) -> nn.Module:
        input_shapes = list(input_shapes)
        return cls(*input_shapes, output_shape=output_shape)

class AblationMLPModel(nn.Module):
    """a simple MLP model for CMTFF ablation experiment."""
    _extractor_key = "extractor"
    _embedding_size = 24

    def __init__(self, *input_shapes, output_shape):
        super().__init__()
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)

        # Extractor
        for index, input_shape in enumerate(input_shapes):
            # Equals to: 1 * elem0 * elem1 * ... * elem_n
            input_dim = reduce(mul, [1] + list(input_shape))
            layer = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=self._embedding_size),
                nn.Sigmoid()
            )

            self.__setattr__(f"{self._extractor_key}{index}", layer)

        # classifier
        self.classifier = nn.Linear(
            in_features=len(input_shapes) * self._embedding_size,
            out_features=output_shape[-1]
        )

    def forward(self, *modal_datas, need_extract_feat=False):
        flatten_modal_datas = [self.flatten_layer(modal_data) for modal_data in modal_datas]
        extracted_feats: [torch.Tensor] = []
        for index, modal_data in enumerate(flatten_modal_datas):
            layer: nn.Module = self.__getattr__(f"{self._extractor_key}{index}")
            extracted_feats.append(layer(modal_data))

        concat_feature = torch.cat(tensors=extracted_feats, axis=1)
        output = self.classifier(concat_feature)

        if need_extract_feat:
            return concat_feature, output
        else:
            return output

    @classmethod
    def instance(cls, *input_shapes: Tuple[Tuple[int]], output_shape: Tuple[int]) -> nn.Module:
        input_shapes = list(input_shapes)
        return cls(*input_shapes, output_shape=output_shape)


class AblationTransformerModel(nn.Module):
    """a Transformer based model for CMTFF ablation experiment."""
    _extractor_key = "extractor"
    embedding_size = 24

    def __init__(self, *input_shapes, output_shape):
        super().__init__()

        # Extractor
        for index, input_shape in enumerate(input_shapes):
            extractor = FeatureExtractor(input_shape=input_shape, embedding_size=self.embedding_size)
            self.__setattr__(f"{self._extractor_key}{index}", extractor)

        # classifier
        self.classifier = nn.Linear(in_features=len(input_shapes) * self.embedding_size, out_features=output_shape[-1])

    def forward(self, *modal_datas, need_extract_feat=False):
        extracted_feats: [torch.Tensor] = []
        for index, modal_data in enumerate(modal_datas):
            layer: nn.Module = self.__getattr__(f"{self._extractor_key}{index}")
            extracted_feats.append(layer(modal_data))

        concat_feature = torch.cat(tensors=extracted_feats, axis=1)
        output = self.classifier(concat_feature)

        if need_extract_feat:
            return concat_feature, output
        else:
            return output

    @classmethod
    def instance(cls, *input_shapes: Tuple[Tuple[int]], output_shape: Tuple[int]) -> nn.Module:
        input_shapes = list(input_shapes)
        return cls(*input_shapes, output_shape=output_shape)
