import logging
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.metrics.core import accuracy_top_k
from gate.models.backbones import GATEncoder
from gate.models.core import SourceModalityConfig, TargetModalityConfig
from gate.models.adapters import BaseAdapterModule

logger = logging.getLogger(__name__)


@configurable(
    group="adapter",
    name="backbone-with-linear-single-classifier",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
class BackboneWithLinearClassification(BaseAdapterModule):
    def __init__(
        self,
        encoder: GATEncoder,
        num_classes: int,
        allow_on_model_metric_computation: bool = True,
        freeze_encoder: bool = False,
        use_stem_instance_norm: bool = False,
    ):
        super().__init__(
            freeze_encoder=freeze_encoder,
            encoder=encoder,
            use_stem_instance_norm=use_stem_instance_norm,
        )

        num_in_features = self.encoder.num_in_features_image
        logger.info(f"Building linear layer with {num_in_features} features.")
        self.num_classes = num_classes
        self.allow_on_model_metric_computation = (
            allow_on_model_metric_computation
        )
        if isinstance(num_classes, int):
            self.linear = nn.Linear(num_in_features, num_classes)
        elif isinstance(num_classes, list):
            self.linear = nn.ModuleList(
                [nn.Linear(num_in_features, n) for n in num_classes]
            )
        elif isinstance(num_classes, dict):
            self.linear = nn.ModuleDict(
                {
                    key: nn.Linear(num_in_features, n)
                    for key, n in num_classes.items()
                }
            )
        elif isinstance(num_classes, DictConfig):
            self.linear = nn.ModuleDict(
                {
                    key: nn.Linear(num_in_features, n)
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
            "labels": (
                torch.randint(0, self.num_classes, (1,))
                if isinstance(self.num_classes, int)
                else {
                    key: torch.randint(0, n, (1,))
                    for key, n in self.num_classes.items()
                }
            ),
        }
        if torch.cuda.device_count() > 1:
            self.linear = self.linear.to(torch.cuda.current_device())
            dummy_batch = {
                k: v.to(torch.cuda.current_device())
                for k, v in dummy_batch.items()
            }

            if hasattr(self, "stem_instance_norm"):
                self.stem_instance_norm = self.stem_instance_norm.to(
                    torch.cuda.current_device()
                )

        _ = self(**dummy_batch)

    @property
    def encoder_transforms(self):
        return self.encoder.get_transforms()

    @property
    def modality_config(self):
        return TargetModalityConfig(image=[SourceModalityConfig(image=True)])

    @ensemble_marker
    def compute_loss_and_metrics_multi_class(self, logits_dict, labels):
        output_dict = {}
        overall_loss = []
        overall_accuracy_top_1 = []
        overall_accuracy_top_5 = []
        for class_type in logits_dict.keys():
            temp_logits = logits_dict[class_type]
            temp_labels = labels[class_type]
            temp_labels = torch.tensor(temp_labels).to(temp_logits.device)
            loss = F.cross_entropy(temp_logits, temp_labels)
            accuracy_top_1 = accuracy_top_k(temp_logits, temp_labels, k=1)
            accuracy_top_5 = accuracy_top_k(
                temp_logits,
                temp_labels,
                k=min(5, self.num_classes[class_type]),
            )

            output_dict[f"loss_{class_type}"] = loss
            output_dict[f"accuracy_top_1_{class_type}"] = accuracy_top_1
            output_dict[
                f"accuracy_top_{min(5, self.num_classes[class_type])}_{class_type}"
            ] = accuracy_top_5
            overall_loss.append(loss)
            overall_accuracy_top_1.append(accuracy_top_1)
            overall_accuracy_top_5.append(accuracy_top_5)

        output_dict["loss"] = torch.mean(torch.stack(overall_loss))
        output_dict["accuracy_top_1"] = torch.mean(
            torch.stack(overall_accuracy_top_1)
        )
        output_dict["accuracy_top_5"] = torch.mean(
            torch.stack(overall_accuracy_top_5)
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
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss_and_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.use_stem_instance_norm:
            image = self.stem_instance_norm(image)
        if image is not None:
            x = self.encoder(image=image)["image"]["features"]

        if text is not None:
            x = self.encoder(text=text)["text"]["features"]

        if audio is not None:
            x = self.encoder(audio=audio)["audio"]["features"]

        if video is not None:
            x = self.encoder(video=video)["video"]["features"]

        if isinstance(self.linear, nn.ModuleDict):
            logits_dict = {}
            labels_dict = {}
            for class_type in self.linear.keys():
                temp_labels = labels[class_type]

                logits_dict[class_type] = self.linear[class_type](x)
                labels_dict[class_type] = temp_labels
            output_dict = {"logits": logits_dict, "labels": labels_dict}
        else:
            output_dict = {
                "logits": self.linear(x),
                "labels": labels,
            }

        if (
            labels is not None
            and return_loss_and_metrics
            and self.allow_on_model_metric_computation
        ):
            output_dict |= self.compute_loss_and_metrics(
                logits=output_dict["logits"], labels=output_dict["labels"]
            )

        return output_dict

    def adapter_transforms(self, inputs: Union[Dict, Any]):
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
