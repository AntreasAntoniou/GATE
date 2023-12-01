from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models.backbones import GATEncoder
from gate.models.core import SourceModalityConfig, TargetModalityConfig
from gate.models.task_adapters import BaseModule


@configurable(
    group="adapter",
    name="multi-class-with-linear-classifier",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
class MultiClassBackboneWithLinear(BaseModule):
    def __init__(
        self,
        encoder: GATEncoder,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.linear = nn.Linear(encoder.num_in_features_image, num_classes)
        self.classes = [f"class{idx}" for idx in range(num_classes)]

        self.build()

    def build(self):
        dummy_batch = {
            "image": torch.randn(
                1, 3, self.encoder.image_shape[0], self.encoder.image_shape[1]
            ),
            "labels": torch.randint(0, self.num_classes, (1,)),
        }
        _ = self(**dummy_batch)

    @property
    def encoder_transforms(self):
        return self.encoder.get_transforms()

    @property
    def modality_config(self):
        return TargetModalityConfig(image=[SourceModalityConfig(image=True)])

    @ensemble_marker
    def compute_multi_class_loss(self, logits, targets):
        opt_loss = F.binary_cross_entropy_with_logits(logits, targets)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        loss = loss.detach()
        logits = logits.detach()

        metrics = {
            "mixed_loss": loss.mean(),
            "loss": opt_loss,
            "predictions": logits,
            "targets": targets,
        }

        for c_idx, class_name in enumerate(self.classes):
            metrics[f"{class_name}-loss"] = loss[:, c_idx].mean()

        return metrics

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if image is not None:
            x = self.encoder(image=image)["image"]["features"]

        if text is not None:
            x = self.encoder(text=text)["text"]["features"]

        if audio is not None:
            x = self.encoder(audio=audio)["audio"]["features"]

        if video is not None:
            x = self.encoder(video=video)["video"]["features"]

        x = self.linear(x)

        if return_loss and labels is not None:
            loss = self.compute_multi_class_loss(x, labels)
            return loss

        return x

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
