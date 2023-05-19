from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.models.task_adapters import BaseModule
from gate.metrics import accuracy_top_k


class BackboneWithLinear(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        num_clip_features,
        num_classes: int,
        modality: str,
    ):
        super().__init__()
        self.model = model
        self.modality = modality
        self.linear = nn.Linear(num_clip_features, num_classes)
        self.num_classes = num_classes

    def compute_metrics(self, logits, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
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

        if labels is not None:
            return self.compute_metrics(x, labels)

        return x
