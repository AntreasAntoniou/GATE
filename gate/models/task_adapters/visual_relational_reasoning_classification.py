from typing import Dict, Optional
from _pytest.stash import D

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.models.task_adapters import BaseModule


class DuoModalFusionModel(BaseModule):
    def __init__(
        self,
        modality_a_model: nn.Module,
        modality_b_model: nn.Module,
        modality_a_identifier: str,
        modality_b_identifier: str,
        modality_a_num_features: int,
        modality_b_num_features: int,
        projection_num_features: Optional[int] = None,
        dropout_fusion_prob: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.modality_a_model = modality_a_model
        self.modality_b_model = modality_b_model

        self.modality_a_identifier = modality_a_identifier
        self.modality_b_identifier = modality_b_identifier

        self.projection_num_features = projection_num_features

        self.temperature_parameter = nn.Parameter(torch.tensor(1.0))

        if self.projection_num_features is not None:
            self.modality_a_linear = nn.Linear(
                modality_a_num_features, projection_num_features, bias=False
            )
            self.modality_b_linear = nn.Linear(
                modality_b_num_features, projection_num_features, bias=False
            )

        self.fusion_in_features = (
            2 * self.projection_num_features
            if self.projection_num_features is not None
            else modality_a_num_features + modality_b_num_features
        )
        # print(self.fusion_in_features, dropout_fusion_prob, num_classes)
        self.fusion_post_processing = nn.Sequential(
            nn.Linear(self.fusion_in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout_fusion_prob),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout_fusion_prob),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # check that only two modalities are passed
        modalities = [image, text, audio, video]
        non_none_modalities = [
            modality for modality in modalities if modality is not None
        ]

        if len(non_none_modalities) != 2:
            raise ValueError(
                f"Exactly two modalities must be provided. You provided {len(non_none_modalities)}"
            )

        modality_a_features = None
        modality_b_features = None

        if image is not None:
            if self.modality_a_identifier == "image":
                modality_a_features = self.modality_a_model(image=image)[
                    self.modality_a_identifier
                ]["features"]
            elif self.modality_b_identifier == "image":
                modality_b_features = self.modality_b_model(image=image)[
                    self.modality_b_identifier
                ]["features"]

        if text is not None:
            if self.modality_a_identifier == "text":
                modality_a_features = self.modality_a_model(text=text)[
                    self.modality_a_identifier
                ]["features"]
            elif self.modality_b_identifier == "text":
                modality_b_features = self.modality_b_model(text=text)[
                    self.modality_b_identifier
                ]["features"]
        if audio is not None:
            if self.modality_a_identifier == "audio":
                modality_a_features = self.modality_a_model(audio=audio)[
                    self.modality_a_identifier
                ]["features"]
            elif self.modality_b_identifier == "audio":
                modality_b_features = self.modality_b_model(audio=audio)[
                    self.modality_b_identifier
                ]["features"]

        if video is not None:
            if self.modality_a_identifier == "video":
                modality_a_features = self.modality_a_model(video=video)[
                    self.modality_a_identifier
                ]["features"]
            elif self.modality_b_identifier == "video":
                modality_b_features = self.modality_b_model(video=video)[
                    self.modality_b_identifier
                ]["features"]

        if self.projection_num_features is not None:
            modality_a_features = self.modality_a_linear(modality_a_features)
            modality_b_features = self.modality_b_linear(modality_b_features)

        # Fusion of the two modalities and post processing

        fused_features = torch.cat(
            [modality_a_features, modality_b_features], dim=1
        )

        logits = self.fusion_post_processing(fused_features)
        output_dict = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output_dict["loss"] = loss
            output_dict["accuracy"] = (
                (logits.detach().argmax(dim=1) == labels).float().mean()
            )

        return output_dict
