import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

from gate.metrics.core import accuracy_top_k
from gate.models.core import reinit
from gate.models.task_adapters import BaseModule

logger = logging.getLogger(__name__)


class PositionalEncoding(BaseModule):
    def __init__(self, has_fixed_context_length: bool = False):
        super().__init__()
        self.has_fixed_context_length = has_fixed_context_length
        self.register_buffer("cached_pe", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_len = x.shape[1]
        d_model = x.shape[2]

        # Check if cached positional encoding can be reused
        if self.has_fixed_context_length and self.cached_pe is not None:
            if (
                self.cached_pe.shape[1] == max_len
                and self.cached_pe.shape[2] == d_model
            ):
                x += self.cached_pe[:, :max_len, :]
                return x

        # Calculate positional encoding
        position = (
            torch.arange(max_len, dtype=torch.float).unsqueeze(1).to(x.device)
        )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        ).to(x.device)
        pe = torch.zeros(1, max_len, d_model).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        if self.has_fixed_context_length:
            self.cached_pe = pe.clone()

        x += pe
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x + self.pos_encoder(x)
        x = self.transformer(x)[:, -1, :]  # take the last frame
        raw_features = self.transformer(x)
        features = self.output_norm(x)
        return {"features": features, "raw_features": raw_features}


def get_classification_metrics_fn_dict(num_classes: int):
    accuracy_top_1 = lambda logits, labels: accuracy_top_k(logits, labels, k=1)
    accuracy_top_5 = lambda logits, labels: accuracy_top_k(
        logits, labels, k=min(5, num_classes)
    )

    loss = lambda logits, labels: F.cross_entropy(logits, labels)

    return {
        "accuracy_top_1": accuracy_top_1,
        "accuracy_top_5": accuracy_top_5,
        "loss": loss,
    }


def get_regression_metrics_fn_dict():
    mse_loss = lambda logits, labels: F.mse_loss(logits, labels)
    mae_loss = lambda logits, labels: F.l1_loss(logits, labels)

    return {
        "mse_loss": mse_loss,
        "mae_loss": mae_loss,
        "loss": mae_loss,
    }


class BackboneWithTemporalTransformerAndLinear(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        num_backbone_features: int,
        num_classes: int,
        metric_fn_dict: Optional[Dict] = None,
    ):
        """Initialize the BackboneWithTemporalTransformerAndLinear module.

        Args:
            model (nn.Module): The backbone neural network model.
            num_backbone_features (int): Number of features output by the backbone model.
            num_classes (int): Number of classes for classification.
            metric_fn_dict (Optional[Dict], optional): Dictionary of metric functions. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.temporal_encoder = VariableSequenceTransformerEncoder(
            d_model=num_backbone_features,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            num_layers=6,
        )
        self.linear = nn.Linear(num_backbone_features, num_classes)
        self.num_classes = num_classes
        self.metric_fn_dict = metric_fn_dict or {}

    def init_weights(self):
        """Initialize the weights of the model."""
        # Assuming `reinit` is a function that initializes the weights
        reinit(self)

    def compute_loss_and_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict:
        """Compute loss and other metrics.

        Args:
            logits (torch.Tensor): The logits output by the model.
            labels (torch.Tensor): The true labels.

        Returns:
            Dict: Dictionary containing computed metrics.
        """
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).to(logits.device)

        output_metric_dict = {
            key: fn(logits, labels) for key, fn in self.metric_fn_dict.items()
        }
        return output_metric_dict

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        video: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss_and_metrics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_dict (Optional[Dict], optional): Input data as a dictionary. Defaults to None.
            video (Optional[torch.Tensor], optional): Video data. Defaults to None.
            labels (Optional[torch.Tensor], optional): True labels. Defaults to None.
            return_loss_and_metrics (bool, optional): Whether to return loss and metrics. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the output logits and optionally the loss and metrics.
        """
        x = self._prepare_input(input_dict, video)
        b, s = x["video"].shape[:2]
        x = x["video"].view(-1, *x["video"].shape[2:])
        x = self._process_through_backbone(x)
        x = x.view(b, s, -1)
        x = self._process_through_temporal_encoder(x)
        logits = self.linear(x)

        output_dict = {"logits": logits}
        if return_loss_and_metrics and labels is not None:
            output_dict.update(
                self.compute_loss_and_metrics(logits=logits, labels=labels)
            )

        return output_dict

    def _prepare_input(
        self, input_dict: Optional[Dict], video: Optional[torch.Tensor]
    ) -> Dict:
        """Prepare the input data for processing.

        Args:
            input_dict (Optional[Dict]): Input data as a dictionary.
            video (Optional[torch.Tensor]): Video data.

        Returns:
            Dict: Prepared input data.
        """
        if input_dict is not None:
            return input_dict
        if video is not None:
            return {"video": video}

    def _process_through_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input through the backbone model.

        Args:
            x (Dict): Input data.

        Returns:
            torch.Tensor: Processed data.
        """
        input_shape = x.shape
        if len(input_shape) == 5:
            x = x.view(-1, *input_shape[-3:])

        x = self.model(video=x)["video"]
        return x["features"]

    def _process_through_temporal_encoder(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Process the input through the temporal encoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Processed data.
        """
        return self.temporal_encoder(x)["features"]
