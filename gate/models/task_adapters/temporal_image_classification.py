import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from accelerate import Accelerator
from rich import print

from gate.boilerplate.utils import get_logger


logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        max_len = x.shape[1]
        d_model = x.shape[2]
        position = torch.arange(max_len).unsqueeze(1).to(x.device)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        ).to(x.device)
        pe = torch.zeros(1, max_len, d_model).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term).to(x.device)
        pe[0, :, 1::2] = torch.cos(position * div_term).to(x.device)
        x = x + pe[: x.size(0)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        batch_first: bool = True,
        norm_first: bool = True,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.activation = activation

        self.pos_encoder = PositionalEncoding()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            encoder_layer=transformer_layer,
            norm=nn.LayerNorm(d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encoder(x)
        x = self.transformer(x)[:, -1, :]  # take the last frame
        raw_features = self.transformer(x)
        features = self.output_norm(x)
        return {"features": features, "raw_features": raw_features}


class BackboneWithTemporalTransformerAndLinear(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_backbone_features: int,
        num_classes: int,
        modality: str,
    ):
        super().__init__()
        self.model = model
        self.modality = modality
        self.temporal_encoder = TransformerEncoder(
            d_model=num_backbone_features,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            num_layers=6,
        )
        self.linear = nn.Linear(num_backbone_features, num_classes)

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
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

        x = self.temporal_encoder(x)["features"]
        x = self.linear(x)
        return x
