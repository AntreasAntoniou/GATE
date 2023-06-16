from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.auto_augment import rand_augment_transform

from gate.metrics import accuracy_top_k
from gate.models.task_adapters import BaseModule


class BackboneWithLinear(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        num_clip_features,
        num_classes: int,
        modality: str,
        allow_on_model_metric_computation: bool = True,
    ):
        super().__init__()
        self.model = model
        self.modality = modality
        self.linear = nn.LazyLinear(num_classes)
        self.num_classes = num_classes
        self.allow_on_model_metric_computation = (
            allow_on_model_metric_computation
        )

    def compute_loss_and_metrics(self, logits, labels):
        if not self.allow_on_model_metric_computation:
            return {}

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

        x = self.linear(x)

        if (
            labels is not None
            and return_loss_and_metrics
            and self.allow_on_model_metric_computation
        ):
            return self.compute_loss_and_metrics(x, labels) | {"logits": x}
        else:
            return {"logits": x}
