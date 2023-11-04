import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from gate.boilerplate.decorators import ensemble_marker
from gate.metrics.core import accuracy_top_k
from gate.models.task_adapters import BaseModule

logger = logging.getLogger(__name__)


class BackboneWithLinear(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        num_in_features,
        num_classes: int,
        modality: str,
        allow_on_model_metric_computation: bool = True,
    ):
        super().__init__()
        self.model = model
        self.modality = modality
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

    @ensemble_marker
    def compute_loss_and_metrics_multi_class(self, logits_dict, labels):
        output_dict = {}
        overall_loss = []
        overall_accuracy_top_1 = []
        for class_type in logits_dict.keys():
            temp_logits = logits_dict[class_type]
            temp_labels = labels[class_type]
            temp_labels = torch.tensor(temp_labels).to(temp_logits.device)
            loss = F.cross_entropy(temp_logits, temp_labels)
            accuracy_top_1 = accuracy_top_k(temp_logits, temp_labels, k=1)

            output_dict[f"loss_{class_type}"] = loss
            output_dict[f"accuracy_top_1_{class_type}"] = accuracy_top_1
            overall_loss.append(loss)
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
        input_dict: Optional[Dict] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss_and_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if input_dict is not None:
            x = self.model(**input_dict)[self.modality]["features"]

        if image is not None:
            x = self.model(image=image)["image"]["features"]

        if text is not None:
            x = self.model(text=text)["text"]["features"]

        if audio is not None:
            x = self.model(audio=audio)["audio"]["features"]

        if video is not None:
            x = self.model(video=video)["video"]["features"]

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
