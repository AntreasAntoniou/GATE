import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.metrics.core import accuracy_top_k
from gate.models.backbones import GATEncoder, reinit
from gate.models.core import SourceModalityConfig, TargetModalityConfig
from gate.models.task_adapters import BaseAdapterModule

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
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


class VariableSequenceTransformerEncoder(nn.Module):
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
        raw_features = x
        features = self.output_norm(x)
        return {
            "features": features,
            "raw_features": raw_features,
            "features": features,
        }


class ClassificationMetrics:
    """
    A class for computing classification metrics such as accuracy and loss.

    Args:
        num_classes (int): The number of classes in the classification task.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def accuracy_top_1(self, logits, labels):
        """
        Computes the top-1 accuracy given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            float: The top-1 accuracy.
        """
        return accuracy_top_k(logits, labels, k=1)

    def accuracy_top_5(self, logits, labels):
        """
        Computes the top-5 accuracy given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            float: The top-5 accuracy.
        """
        return accuracy_top_k(logits, labels, k=min(5, self.num_classes))

    def loss(self, logits, labels):
        """
        Computes the cross-entropy loss given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The cross-entropy loss.
        """
        return F.cross_entropy(logits, labels)

    def __call__(self, logits, labels):
        """
        Computes the classification metrics given the logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        return {
            "accuracy_top_1": self.accuracy_top_1(logits, labels),
            "accuracy_top_5": self.accuracy_top_5(logits, labels),
            "loss": self.loss(logits, labels),
        }


class RegressionMetrics:
    """
    A class for computing regression metrics such as mean squared error (MSE) and mean absolute error (MAE).
    """

    def mse_loss(self, logits, labels):
        """
        Computes the mean squared error (MSE) loss between the predicted logits and the true labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The MSE loss.
        """
        return F.mse_loss(logits.view(-1), labels.view(-1))

    def mae_loss(self, logits, labels):
        """
        Computes the mean absolute error (MAE) loss between the predicted logits and the true labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The MAE loss.
        """
        return F.l1_loss(logits.view(-1), labels.view(-1))

    def __call__(self, logits, labels):
        """
        Computes the MSE and MAE losses between the predicted logits and the true labels.

        Args:
            logits (torch.Tensor): The predicted logits.
            labels (torch.Tensor): The true labels.

        Returns:
            dict: A dictionary containing the MSE loss, MAE loss, and total loss.
        """
        return {
            "mse_loss": self.mse_loss(logits, labels).detach(),
            "mae_loss": self.mae_loss(logits, labels).detach(),
            "loss": self.mae_loss(
                logits, labels
            ),  # Assuming you want the 'loss' key to map to mae_loss
        }


def get_metric_fn(metric_type, num_classes):
    metrics = {
        Metrics.CLASSIFICATION: ClassificationMetrics(num_classes=num_classes),
        Metrics.REGRESSION: RegressionMetrics(),
    }

    return metrics[metric_type]


@dataclass
class Metrics:
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    get_metric_fn = get_metric_fn


@configurable(
    group="adapter",
    name="temporal-classification",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
class BackboneWithTemporalTransformerAndLinear(BaseAdapterModule):
    def __init__(
        self,
        encoder: GATEncoder,
        num_classes: int,
        metric_type: str = Metrics.CLASSIFICATION,
        temporal_transformer_nhead: int = 8,
        temporal_transformer_dim_feedforward: int = 2048,
        temporal_transformer_dropout: float = 0.0,
        temporal_transformer_num_layers: int = 6,
        freeze_encoder: bool = False,
        use_stem_instance_norm: bool = False,
    ):
        """Initialize the BackboneWithTemporalTransformerAndLinear module.

        Args:
            model (nn.Module): The backbone neural network model.
            num_backbone_features (int): Number of features output by the backbone model.
            num_classes (int): Number of classes for classification.
            metric_fn_dict (Optional[Dict], optional): Dictionary of metric functions. Defaults to None.
        """
        super().__init__(
            encoder=encoder,
            freeze_encoder=freeze_encoder,
            use_stem_instance_norm=use_stem_instance_norm,
        )
        self.temporal_encoder = VariableSequenceTransformerEncoder(
            d_model=encoder.num_in_features_image,
            nhead=temporal_transformer_nhead,
            dim_feedforward=temporal_transformer_dim_feedforward,
            dropout=temporal_transformer_dropout,
            num_layers=temporal_transformer_num_layers,
        )
        self.linear = nn.Linear(
            in_features=encoder.num_in_features_image, out_features=num_classes
        )
        self.num_classes = num_classes
        self.metric_fn_dict = Metrics.get_metric_fn(
            metric_type, num_classes=num_classes
        )
        self.build()

    def build(self):
        dummy_batch = {
            "video": torch.randn(
                1,
                1,
                3,
                self.encoder.image_shape[0],
                self.encoder.image_shape[1],
            ),
            "labels": torch.randint(0, self.num_classes, (1,)),
        }
        if torch.cuda.device_count() > 1:
            self.linear = self.linear.to(torch.cuda.current_device())
            self.temporal_encoder = self.temporal_encoder.to(
                torch.cuda.current_device()
            )
            dummy_batch = {
                k: v.to(torch.cuda.current_device())
                for k, v in dummy_batch.items()
            }

            if hasattr(self, "stem_instance_norm"):
                self.stem_instance_norm = self.stem_instance_norm.to(
                    torch.cuda.current_device()
                )

        _ = self(**dummy_batch)

    @property
    def modality_config(self):
        return TargetModalityConfig(video=[SourceModalityConfig(video=True)])

    @property
    def encoder_transforms(self):
        return self.encoder.get_transforms()

    def init_weights(self):
        """Initialize the weights of the model."""
        # Assuming `reinit` is a function that initializes the weights
        reinit(self)

    @ensemble_marker
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

        output_metric_dict = self.metric_fn_dict(logits, labels)
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

        if self.use_stem_instance_norm:
            x = self.stem_instance_norm(x)

        x = self.encoder(video=x)["video"]
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

    def adapter_transforms(self, inputs: Union[Dict, Any]):
        if isinstance(inputs, dict):
            output_dict = {
                "video": self.encoder_transforms["video"](inputs["video"]),
            }
        else:
            output_dict = {"video": self.encoder_transforms["video"](inputs)}

        inputs.update(output_dict)
        output_dict = inputs

        return output_dict
