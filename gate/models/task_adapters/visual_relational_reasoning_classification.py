from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.metrics.core import accuracy_top_k
from gate.models.backbones import GATEncoder
from gate.models.core import SourceModalityConfig, TargetModalityConfig
from gate.models.task_adapters import BaseModule
from gate.models.task_adapters.temporal_image_classification import (
    VariableSequenceTransformerEncoder,
)
from gate.models.task_adapters.utils import reinit


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
class DuoModalFusionModel(BaseModule):
    def __init__(
        self,
        encoder: GATEncoder,
        projection_num_features: int = 512,
        dropout_fusion_prob: float = 0.0,
        num_classes: Union[List[int], int, Dict[str, int]] = 10,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection_num_features = projection_num_features

        self.temperature_parameter = nn.Parameter(torch.tensor(1.0))

        self.image_linear = nn.Linear(
            self.encoder.num_in_features_image,
            projection_num_features,
            bias=False,
        )
        self.text_image = nn.Linear(
            self.encoder.num_in_features_text,
            projection_num_features,
            bias=False,
        )

        self.fusion_in_features = self.projection_num_features
        self.image_instance_norm = nn.InstanceNorm2d(
            3, affine=True, track_running_stats=False
        )
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
                1, 3, self.encoder.image_shape[0], self.encoder.image_shape[1]
            ),
            "labels": torch.randint(0, self.num_classes, (1,))
            if isinstance(self.num_classes, int)
            else {
                key: torch.randint(0, self.num_classes[key], (1,))
                for key in self.num_classes.keys()
            }
            if isinstance(self.num_classes, dict)
            else [
                torch.randint(0, item)
                for item in self.num_classes
                if isinstance(item, list)
            ],
        }
        _ = self(**dummy_batch)

    @ensemble_marker
    def compute_loss_and_metrics_multi_class(self, logits_dict, labels):
        output_dict = {}
        overall_loss = []
        overall_accuracy_top_1 = []
        for answer in logits_dict.keys():
            temp_logits = logits_dict[answer]
            temp_labels = labels[answer]
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
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        answer_type: Optional[str] = None,
        question_family_idx: Optional[int] = None,
        return_loss_and_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # check that only two modalities are passed
        modalities = [image, text]
        non_none_modalities = [
            modality for modality in modalities if modality is not None
        ]

        if len(non_none_modalities) != 2:
            raise ValueError(
                f"Exactly two modalities must be provided. You provided {len(non_none_modalities)}"
            )

        image_features = None
        text_features = None

        if image is not None:
            image = self.image_instance_norm(image)
            image_features = self.encoder(image=image)["image"]["raw_features"]

        if text is not None:
            text_features = self.encoder(text=text)["text"]["raw_features"]

        image_features = self.image_linear(
            image_features.reshape(-1, image_features.shape[-1])
        ).reshape(
            image_features.shape[0],
            image_features.shape[1],
            self.projection_num_features,
        )

        text_features = self.text_image(
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
                    item
                    for item in range(len(answer_type))
                    if answer_type[item] == answer
                ]
                temp_features = features[answer_specific_idx]
                temp_labels = labels[answer_specific_idx]

                logits_dict[answer] = self.classifier[answer](temp_features)
                labels_dict[answer] = temp_labels
                output_dict = {"logits": logits_dict, "labels": labels_dict}
        else:
            output_dict = {
                "logits": self.classifier(features),
                "labels": labels,
            }

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
        reinit(self)

    def adapter_transforms(self, inputs: dict):
        output_dict = {}

        if "image" in inputs:
            output_dict["image"] = self.encoder_transforms["image"](
                inputs["image"]
            )

        if "text" in inputs:
            output_dict["text"] = self.encoder_transforms["text"](
                inputs["text"]
            )

        if "labels" in inputs:
            output_dict["labels"] = inputs["labels"]

        return output_dict
