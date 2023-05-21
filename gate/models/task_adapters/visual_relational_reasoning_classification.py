from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.models.task_adapters import BaseModule
from gate.models.task_adapters.temporal_image_classification import (
    VariableSequenceTransformerEncoder,
)


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


class DuoModalFusionModel(BaseModule):
    def __init__(
        self,
        modality_a_model: nn.Module,
        modality_b_model: nn.Module,
        modality_a_identifier: str,
        modality_b_identifier: str,
        modality_a_num_features: int,
        modality_b_num_features: int,
        projection_num_features: int = 512,
        dropout_fusion_prob: float = 0.0,
        num_classes: int = 10,
    ):
        super().__init__()
        self.modality_a_model = modality_a_model
        self.modality_b_model = modality_b_model

        self.modality_a_identifier = modality_a_identifier
        self.modality_b_identifier = modality_b_identifier

        self.projection_num_features = projection_num_features

        self.temperature_parameter = nn.Parameter(torch.tensor(1.0))

        self.modality_a_linear = nn.Linear(
            modality_a_num_features, projection_num_features, bias=False
        )
        self.modality_b_linear = nn.Linear(
            modality_b_num_features, projection_num_features, bias=False
        )

        self.fusion_in_features = self.projection_num_features
        # print(self.fusion_in_features, dropout_fusion_prob, num_classes)
        self.fusion_post_processing = VariableSequenceTransformerEncoder(
            d_model=self.fusion_in_features,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            num_layers=4,
        )
        self.classifier = nn.Linear(self.fusion_in_features, num_classes)

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
        text[text == -1] = self.modality_a_model.tokenizer.eos_token_id

        if image is not None:
            if self.modality_a_identifier == "image":
                modality_a_features = self.modality_a_model(image=image)[
                    self.modality_a_identifier
                ]["raw_features"]
            elif self.modality_b_identifier == "image":
                modality_b_features = self.modality_b_model(image=image)[
                    self.modality_b_identifier
                ]["raw_features"]

        if text is not None:
            if self.modality_a_identifier == "text":
                text[text == -1] = self.modality_a_model.tokenizer.eos_token_id
                modality_a_features = self.modality_a_model(text=text)[
                    self.modality_a_identifier
                ]["raw_features"]
            elif self.modality_b_identifier == "text":
                modality_b_features = self.modality_b_model(text=text)[
                    self.modality_b_identifier
                ]["raw_features"]
        if audio is not None:
            if self.modality_a_identifier == "audio":
                modality_a_features = self.modality_a_model(audio=audio)[
                    self.modality_a_identifier
                ]["raw_features"]
            elif self.modality_b_identifier == "audio":
                modality_b_features = self.modality_b_model(audio=audio)[
                    self.modality_b_identifier
                ]["raw_features"]

        if video is not None:
            if self.modality_a_identifier == "video":
                modality_a_features = self.modality_a_model(video=video)[
                    self.modality_a_identifier
                ]["raw_features"]
            elif self.modality_b_identifier == "video":
                modality_b_features = self.modality_b_model(video=video)[
                    self.modality_b_identifier
                ]["raw_features"]

        modality_a_features = self.modality_a_linear(
            modality_a_features.view(-1, modality_a_features.shape[-1])
        ).view(
            modality_a_features.shape[0],
            modality_a_features.shape[1],
            self.projection_num_features,
        )
        modality_b_features = self.modality_b_linear(
            modality_b_features.view(-1, modality_b_features.shape[-1])
        ).view(
            modality_b_features.shape[0],
            modality_b_features.shape[1],
            self.projection_num_features,
        )

        # Fusion of the two modalities and post processing

        fused_features = torch.cat(
            [modality_a_features, modality_b_features], dim=1
        )

        features = self.fusion_post_processing(fused_features)["features"]
        logits = self.classifier(features)
        output_dict = {"logits": logits}

        return output_dict
