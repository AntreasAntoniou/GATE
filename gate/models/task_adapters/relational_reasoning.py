import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from omegaconf import DictConfig

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.boilerplate.utils import get_logger
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.metrics.core import accuracy_top_k
from gate.models.backbones import GATEncoder
from gate.models.core import (
    SourceModalityConfig,
    TargetModalityConfig,
    simple_init,
)
from gate.models.task_adapters import BaseAdapterModule
from gate.models.task_adapters.utils.helpers import reinit

logger = get_logger(__name__, set_rich=True)


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, has_fixed_context_length: bool = False):
        super().__init__()
        self.has_fixed_context_length = has_fixed_context_length
        self.register_buffer("cached_pe", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_len = x.shape[1]
        d_model = x.shape[2]

        # Check if cached positional encoding can be reused
        if self.has_fixed_context_length and self.cached_pe is not None:
            if (
                self.cached_pe.shape[1] == max_len
                and self.cached_pe.shape[2] == d_model
            ):
                x += self.cached_pe[:, :max_len, :]
                return x

        # Calculate positional encoding
        position = (
            torch.arange(max_len, dtype=torch.float).unsqueeze(1).to(x.device)
        )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        ).to(x.device)
        pe = torch.zeros(1, max_len, d_model).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        if self.has_fixed_context_length:
            self.cached_pe = pe.clone()

        x += pe
        return x


class VariableSequenceTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        batch_first: bool = True,
        norm_first: bool = True,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.activation = activation

        self.pos_encoder = PositionalEncoding()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            encoder_layer=transformer_layer,
            norm=nn.LayerNorm(d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x + self.pos_encoder(x)
        x = self.transformer(x)[:, -1, :]  # take the last frame
        raw_features = self.transformer(x)
        features = self.output_norm(x)
        return {
            "features": features,
            "raw_features": raw_features,
            "features": features,
        }


class ClassificationMetrics:
    """
    A class for computing classification metrics such as accuracy and loss.

    Args:
        num_classes (int): The number of classes in the classification task.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def accuracy_top_1(self, logits, labels):
        """
        Computes the top-1 accuracy given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            float: The top-1 accuracy.
        """
        return accuracy_top_k(logits, labels, k=1)

    def accuracy_top_5(self, logits, labels):
        """
        Computes the top-5 accuracy given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            float: The top-5 accuracy.
        """
        return accuracy_top_k(logits, labels, k=min(5, self.num_classes))

    def loss(self, logits, labels):
        """
        Computes the cross-entropy loss given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The cross-entropy loss.
        """
        return F.cross_entropy(logits, labels)

    def __call__(self, logits, labels):
        """
        Computes the classification metrics given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        return {
            "accuracy_top_1": self.accuracy_top_1(logits, labels),
            "accuracy_top_5": self.accuracy_top_5(logits, labels),
            "loss": self.loss(logits, labels),
        }


class SkipConnectionModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)

        b, f = x.shape
        _, k = out.shape

        # calculate the padding size
        padding_size = k - f

        # use the F.pad function
        # as we want to pad the last dimension, we provide the padding size as (0, padding_size)
        # (each pair in the padding argument pads the corresponding dimension in the input tensor)
        if padding_size > 0:
            x = F.pad(x, (0, padding_size))

        return out + x


@configurable(
    group="adapter",
    name="relational-reasoning",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
class DuoModalFusionModel(BaseAdapterModule):
    def __init__(
        self,
        encoder: GATEncoder,
        dropout_fusion_prob: float = 0.0,
        num_classes: Union[List[int], int, Dict[str, int]] = 10,
        projection_num_features: int = 512,
        freeze_encoder: bool = False,
        use_stem_instance_norm: bool = False,
    ):
        super().__init__(
            encoder=encoder,
            freeze_encoder=freeze_encoder,
            use_stem_instance_norm=use_stem_instance_norm,
        )

        self.temperature_parameter = nn.Parameter(torch.tensor(1.0))
        self.projection_num_features = projection_num_features

        self.image_linear = nn.Linear(
            self.encoder.num_raw_features_image,
            projection_num_features,
            bias=False,
        )
        self.text_linear = nn.Linear(
            self.encoder.num_raw_features_text,
            projection_num_features,
            bias=False,
        )

        self.fusion_in_features = projection_num_features

        self.fusion_post_processing = VariableSequenceTransformerEncoder(
            d_model=self.fusion_in_features,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout_fusion_prob,
            num_layers=4,
        )
        self.num_classes = num_classes
        if isinstance(num_classes, int):
            self.classifier = nn.Linear(self.fusion_in_features, num_classes)
        elif isinstance(num_classes, list):
            self.classifier = nn.ModuleList(
                [nn.Linear(self.fusion_in_features, n) for n in num_classes]
            )
        elif isinstance(num_classes, dict):
            self.classifier = nn.ModuleDict(
                {
                    key: nn.Linear(self.fusion_in_features, n)
                    for key, n in num_classes.items()
                }
            )
        elif isinstance(num_classes, DictConfig):
            self.classifier = nn.ModuleDict(
                {
                    key: nn.Linear(self.fusion_in_features, n)
                    for key, n in num_classes.items()
                }
            )
        else:
            raise ValueError(
                f"num_classes must be either int, list or dict. You provided {type(num_classes)}"
            )

        self.build()

    def build(self):
        dummy_batch = {
            "image": torch.randn(
                2, 3, self.encoder.image_shape[0], self.encoder.image_shape[1]
            ),
            "text": torch.randint(0, 100, (2, 10)),
            "labels": (
                torch.randint(0, self.num_classes, (2,))
                if isinstance(self.num_classes, int)
                else torch.stack(
                    [
                        torch.randint(value, (1,))
                        for value in list(self.num_classes.values())[:2]
                    ]
                )
            ),
            "answer_type": (
                list(self.num_classes.keys())[:2]
                if not isinstance(self.num_classes, int)
                else None
            ),
        }

        _ = self(**dummy_batch)

    @ensemble_marker
    def compute_loss_and_metrics_multi_class(self, logits_dict, labels):
        output_dict = {}
        overall_loss = []
        overall_accuracy_top_1 = []
        for answer in logits_dict.keys():
            temp_logits = logits_dict[answer]
            temp_labels = labels[answer].view(-1)
            loss = F.cross_entropy(temp_logits, temp_labels, reduction="none")
            accuracy_top_1 = accuracy_top_k(temp_logits, temp_labels, k=1)

            output_dict[f"loss_{answer}"] = torch.mean(loss)
            output_dict[f"accuracy_top_1_{answer}"] = accuracy_top_1
            overall_loss.extend(loss)
            overall_accuracy_top_1.append(accuracy_top_1)

        output_dict["loss"] = torch.mean(torch.stack(overall_loss))
        output_dict["accuracy_top_1"] = torch.mean(
            torch.stack(overall_accuracy_top_1)
        )
        return output_dict

    @ensemble_marker
    def compute_loss_and_metrics_single_class(self, logits, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).to(logits.device)

        accuracy_top_1 = accuracy_top_k(logits, labels, k=1)
        accuracy_top_5 = accuracy_top_k(
            logits, labels, k=min(5, self.num_classes)
        )

        loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "accuracy_top_1": accuracy_top_1,
            "accuracy_top_5": accuracy_top_5,
        }

    @ensemble_marker
    def compute_loss_and_metrics(self, logits, labels):
        if isinstance(logits, dict):
            return self.compute_loss_and_metrics_multi_class(logits, labels)

        else:
            return self.compute_loss_and_metrics_single_class(logits, labels)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        answer_type: Optional[str] = None,
        question_family_idx: Optional[int] = None,
        return_loss_and_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # check that only two modalities are passed

        if self.use_stem_instance_norm:
            image = self.image_instance_norm(image)
        image_features = self.encoder(image=image)["image"]["raw_features"]
        image_features = self.image_linear(
            image_features.reshape(-1, image_features.shape[-1])
        ).reshape(
            image_features.shape[0],
            image_features.shape[1],
            self.projection_num_features,
        )

        text_features = self.encoder(text=text)["text"]["raw_features"]

        text_features = self.text_linear(
            text_features.reshape(-1, text_features.shape[-1])
        ).reshape(
            text_features.shape[0],
            text_features.shape[1],
            self.projection_num_features,
        )

        # Fusion of the two modalities and post processing
        fused_features = torch.cat([image_features, text_features], dim=1)

        features = self.fusion_post_processing(fused_features)["features"]
        logits_dict = {}
        labels_dict = {}
        if isinstance(self.classifier, nn.ModuleDict):
            for answer in self.classifier.keys():
                answer_specific_idx = [
                    idx
                    for idx, item in enumerate(answer_type)
                    if item == answer
                ]
                if len(answer_specific_idx) == 0:
                    continue
                temp_features = features[answer_specific_idx]

                logits_dict[answer] = self.classifier[answer](temp_features)

                if labels is not None:
                    labels_dict[answer] = (
                        labels[answer_specific_idx]
                        if isinstance(labels, torch.Tensor)
                        else [labels[idx] for idx in answer_specific_idx]
                    )

                output_dict = {"logits": logits_dict, "labels": labels_dict}
        else:
            output_dict = {
                "logits": self.classifier(features),
            }
            if labels is not None:
                output_dict["labels"] = labels

        if labels is not None and return_loss_and_metrics:
            output_dict |= self.compute_loss_and_metrics(
                logits=output_dict["logits"], labels=output_dict["labels"]
            )

        return output_dict

    @property
    def modality_config(self):
        return TargetModalityConfig(
            image_text=[SourceModalityConfig(image=True, text=True)]
        )

    @property
    def encoder_transforms(self):
        return self.encoder.get_transforms()

    def init_weights(self):
        """Initialize the weights of the model."""
        # Assuming `reinit` is a function that initializes the weights
        simple_init(self)

    def adapter_transforms(self, inputs: dict):
        if "image" in inputs:
            inputs["image"] = self.encoder_transforms["image"](inputs["image"])

        if "text" in inputs:
            inputs["text"] = self.encoder_transforms["text"](inputs["text"])

        return inputs
