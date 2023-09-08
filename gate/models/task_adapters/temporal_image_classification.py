import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

from gate.boilerplate.utils import get_logger
from gate.metrics.core import accuracy_top_k
from gate.models.core import reinit
from gate.models.task_adapters import BaseModule

logger = get_logger(__name__)


class PositionalEncoding(BaseModule):
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


class VariableSequenceTransformerEncoder(BaseModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        batch_first: bool = True,
        norm_first: bool = True,
        activation: nn.Module = nn.GELU(),
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


class BackboneWithTemporalTransformerAndLinear(BaseModule):
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
        self.temporal_encoder = VariableSequenceTransformerEncoder(
            d_model=num_backbone_features,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            num_layers=6,
        )
        self.linear = nn.Linear(num_backbone_features, num_classes)

    def init_weights(self):
        reinit(self)

    def compute_loss_and_metrics(self, logits, labels):
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
        return_loss_and_metrics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if input_dict is not None:
            x = {self.modality: input_dict[self.modality]}

        if image is not None:
            # e.g. image.shape = [batch_size, seq_len, 3, 224, 224]
            x = {"image": image}

        if text is not None:
            x = {"text": text}

        if audio is not None:
            x = {"audio": audio}

        if video is not None:
            x = {"video": video}

        input_shape = x[self.modality].shape

        if len(input_shape) == 5:
            x = x[self.modality].view(-1, *input_shape[-3:])

        x = self.model(**{self.modality: x})[
            self.modality
        ]  # [batch_size * seq_len, 3, 224, 224] -> model -> [batch_size * seq_len, num_backbone_features]
        x = x["features"].view(
            *input_shape[:2], -1
        )  # [batch_size, seq_len, num_backbone_features]
        x = self.temporal_encoder(x)[
            "features"
        ]  # [batch_size, num_backbone_features]
        x = self.linear(x)  # [batch_size, num_classes]

        output_dict = {"logits": x}

        if return_loss_and_metrics and labels is not None:
            output_dict |= self.compute_loss_and_metrics(
                logits=output_dict["logits"], labels=labels
            )

        return output_dict
