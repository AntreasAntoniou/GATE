from typing import Dict, Optional

import torch.nn as nn
import torch

from gate.models.task_adapters import BaseModule
from gate.models.task_adapters.extras import (
    get_similarities,
)


class DuoModalZeroShotModel(BaseModule):
    def __init__(
        self,
        modality_a_model: nn.Module,
        modality_b_model: nn.Module,
        modality_a_identifier: str,
        modality_b_identifier: str,
        modality_a_num_features: int,
        modality_b_num_features: int,
        projection_num_features: Optional[int] = None,
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

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        return_loss: bool = False,
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

        metrics_dict = get_similarities(
            modality_a_name=self.modality_a_identifier,
            modality_a_features=modality_a_features,
            modality_b_name=self.modality_b_identifier,
            modality_b_features=modality_b_features,
            temperature_parameter=self.temperature_parameter,
            return_loss=return_loss,
        )

        losses_list = [
            value for key, value in metrics_dict.items() if "loss" in key
        ]
        if len(losses_list) > 0:
            loss = torch.mean(torch.stack(losses_list))
            metrics_dict["loss"] = loss
        return metrics_dict
