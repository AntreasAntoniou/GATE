import math
import time
from typing import List, Optional, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.boilerplate.utils import get_logger
from gate.metrics.segmentation import (
    DiceLoss,
    FocalLoss,
    IoUMetric,
    WeightedCrossEntropyLoss,
)

logger = get_logger(__name__)


def has_exact_square_root(s: int) -> bool:
    # Get the size of the second dimension (s)

    # Calculate the square root of s
    root = math.sqrt(s)

    # Check if the square root is an integer
    return root.is_integer()


def optimization_loss(
    logits, labels, ignore_index: int = 0, background_loss_weight: float = 0.0
):
    """
    📝 Optimization Loss
    Args:
        logits: (B, C, H, W)
        labels: (B, 1, H, W)
    """
    dice_loss_fn = DiceLoss(ignore_index=ignore_index)

    dice_loss = dice_loss_fn.forward(logits, labels)

    focal_loss_fn = FocalLoss(ignore_index=ignore_index)

    focal_loss = focal_loss_fn.forward(logits, labels)

    ce_loss_fn = WeightedCrossEntropyLoss(ignore_index=ignore_index)

    ce_loss = ce_loss_fn.forward(logits, labels)

    background_dice_loss_fn = DiceLoss(ignore_index=-1)

    background_dice_loss = background_dice_loss_fn.forward(logits, labels)

    background_focal_loss_fn = FocalLoss(ignore_index=-1)

    background_focal_loss = background_focal_loss_fn.forward(logits, labels)

    background_ce_loss_fn = WeightedCrossEntropyLoss(ignore_index=-1)

    background_ce_loss = background_ce_loss_fn.forward(logits, labels)

    loss = dice_loss + focal_loss
    # 0.1 * ce_loss +
    background_loss = background_dice_loss + background_focal_loss
    # 0.1 * background_ce_loss

    return {
        "loss": loss + background_loss_weight * background_loss,
        "ce_loss": ce_loss,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "background_loss": background_loss,
        "background_dice_loss": background_dice_loss,
        "background_focal_loss": background_focal_loss,
        "background_ce_loss": background_ce_loss,
    }


class ChannelMixerDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        target_image_size: Union[int, tuple],
        hidden_size: int = 256,
        dropout_rate: float = 0.5,
        pre_output_dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_image_size = (
            target_image_size
            if isinstance(target_image_size, int)
            else target_image_size[0]
        )
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.pre_output_dropout_rate = pre_output_dropout_rate
        self.is_built = False

    def build(self, input_list: List[torch.Tensor]):
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bilinear", align_corners=True
        )
        out = torch.cat([self.upsample(x) for x in input_list], dim=1)
        input_shape = out.shape

        norm = (
            nn.LazyInstanceNorm2d
            if len(input_shape) == 4
            else nn.LazyInstanceNorm1d
        )
        conv = nn.LazyConv2d if len(input_shape) == 4 else nn.LazyConv1d
        dropout = nn.Dropout2d if len(input_shape) == 4 else nn.Dropout1d
        act = nn.LeakyReLU

        self.mlp = nn.Sequential(
            conv(out_channels=self.hidden_size, kernel_size=1),
            norm(),
            act(),
            dropout(self.dropout_rate),
            conv(out_channels=self.hidden_size, kernel_size=1),
            norm(self.hidden_size),
            act(),
            dropout(self.dropout_rate),
        )

        self.fuse_features = nn.Sequential(
            conv(out_channels=self.hidden_size, kernel_size=1),
            norm(self.hidden_size),
            act(),
            dropout(self.pre_output_dropout_rate),
        )

        self.final_conv = nn.LazyConv2d(
            out_channels=self.num_classes, kernel_size=1
        )
        self.is_built = True

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        if not self.is_built:
            self.build(input_list)

        input_feature_maps = [self.upsample(x) for x in input_list]
        input_feature_maps = torch.cat(input_feature_maps, dim=1)
        processed_features = self.mlp(input_feature_maps)
        fused_features = self.fuse_features(processed_features)
        class_features = self.final_conv(fused_features)
        return class_features


def upsample_tensor(input_tensor):
    b, c, s = input_tensor.shape
    new_size = math.ceil(math.sqrt(s)) ** 2
    sq_root = int(math.sqrt(new_size))
    new_shape = (b, c, new_size)
    output_tensor = F.upsample(
        input_tensor,
        size=(sq_root * sq_root),
    )
    return output_tensor.view(b, c, sq_root, sq_root)


class TransformerSegmentationDecoderHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        target_image_size: Union[int, tuple],
        hidden_size: int = 256,
        dropout_rate: float = 0.5,
        pre_output_dropout_rate: float = 0.3,
        num_transformer_blocks: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_image_size = (
            target_image_size
            if isinstance(target_image_size, int)
            else target_image_size[0]
        )
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.pre_output_dropout_rate = pre_output_dropout_rate
        self.num_transformer_blocks = num_transformer_blocks
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bilinear", align_corners=True
        )
        self.is_built = False

    def build(self, in_channels: int):
        # Define projection layer
        self.projection_layer = nn.Conv2d(in_channels, self.hidden_size, 1)

        # Define feature fusion and transformer layers
        self.fuse_features = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, 1),
            nn.InstanceNorm2d(self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout2d(self.pre_output_dropout_rate),
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.segmentation_processing_head = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=self.num_transformer_blocks,
            norm=nn.LayerNorm(self.hidden_size),
        )

        self.final_conv = nn.Conv2d(self.hidden_size, self.num_classes, 1)
        self.is_built = True

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        input_feature_maps = [self.upsample(x) for x in input_list]
        input_feature_maps = torch.cat(input_feature_maps, dim=1)

        if not self.is_built:
            self.build(input_feature_maps.shape[1])

        projected_features = self.projection_layer(input_feature_maps)
        fused_features = self.fuse_features(projected_features)

        fused_features = einops.rearrange(
            fused_features, "b c h w -> b (h w) c"
        )

        transformed_features = self.segmentation_processing_head(
            fused_features
        )

        transformed_features = einops.rearrange(
            transformed_features,
            "b (h w) c -> b c h w",
            h=self.target_image_size,
            w=self.target_image_size,
        )
        class_features = self.final_conv(transformed_features)

        return class_features


class SegmentationAdapter(nn.Module):
    def __init__(
        self,
        encoder_model: nn.Module,
        decoder_embed_dim: int = 512,
        num_classes: int = 100,
        background_loss_weight: float = 0.0,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 0,
        target_image_size: tuple = (64, 64),
        decoder_layer_type: str = "simple",
    ):
        super().__init__()

        self.encoder = encoder_model
        self.num_classes = num_classes
        self.class_names = (
            class_names
            if class_names
            else [str(i) for i in range(num_classes)]
        )
        self.ignore_index = ignore_index
        self.decoder_embedding_dimension = decoder_embed_dim

        # Assuming decoder_layer_mapping and other related classes and functions are defined elsewhere in the code
        decoder_layer_mapping = {
            "transformer": TransformerSegmentationDecoderHead,
            "simple": ChannelMixerDecoder,
        }

        self.decoder_head = decoder_layer_mapping[decoder_layer_type](
            input_feature_maps=[],
            num_classes=num_classes,
            target_image_size=target_image_size[0],
            hidden_size=decoder_embed_dim,
        )

        self.iou_metric = IoUMetric(
            num_classes,
            ignore_index,
            {i: name for i, name in enumerate(self.class_names)},
        )
        self.background_loss_weight = background_loss_weight

    def compute_across_set_iou(self):
        metrics = self.iou_metric.compute_metrics()
        self.iou_metric.reset()  # Resetting the metrics after computation
        return metrics

    def forward(self, image, labels: Optional[torch.Tensor] = None):
        features = self.encoder(image)["image"]["per_layer_raw_features"]

        mask_predictions = self.decoder_head(features)

        logits = F.interpolate(
            input=mask_predictions,
            size=(
                256,
                256,
            ),  # Changed to a more common image size for generality; adjust as needed
            mode="bicubic",
            align_corners=True,
        )

        output = {"logits": logits.detach()}

        if labels is not None:
            output.update(self.compute_loss_and_metrics(logits, labels))

        return output

    def compute_loss_and_metrics(self, logits, labels):
        loss_and_metrics = optimization_loss(
            logits,
            labels,
            ignore_index=self.ignore_index,
            background_loss_weight=self.background_loss_weight,  # Assuming optimization_loss is defined elsewhere
        )

        if not self.training:
            preds = torch.argmax(logits, dim=1)
            self.iou_metric.update(preds, labels)

        return loss_and_metrics
