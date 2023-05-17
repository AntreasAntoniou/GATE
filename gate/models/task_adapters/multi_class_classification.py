from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.models.task_adapters import BaseModule


class MultiClassBackboneWithLinear(BaseModule):
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
        self.classes = [f"class{idx}" for idx in range(num_classes)]

    def compute_multi_class_loss(self, logits, targets):
        opt_loss = F.binary_cross_entropy_with_logits(logits, targets)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        loss = loss.detach()
        logits = logits.detach()

        metrics = {"mixed_loss": loss.mean(), "loss": opt_loss, "predictions": logits, "targets": targets}
        
        for c_idx, class_name in enumerate(self.classes):
            metrics[f"{class_name}-loss"] = loss[:, c_idx].mean()
        
        return metrics
        
    
    def forward(
        self,,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        if image is not None:
            x = self.model(image=image)["image"]["features"]

        if text is not None:
            x = self.model(text=text)["text"]["features"]

        if audio is not None:
            x = self.model(audio=audio)["audio"]["features"]

        if video is not None:
            x = self.model(video=video)["video"]["features"]

        x = self.linear(x)
        
        if return_loss and labels is not None:
            loss = self.compute_multi_class_loss(x, labels)
            return loss
        
        return x
