import logging
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.metrics.segmentation import (
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    IoUMetric,
)
from gate.models.backbones import GATEncoder
from gate.models.blocks.segmentation import (
    ChannelMixerDecoder,
    TransformerSegmentationDecoder,
)
from gate.models.core import SourceModalityConfig, TargetModalityConfig
from gate.models.task_adapters.utils import reinit

# from mmseg.evaluation.metrics import IoUMetric as mmsegIoUMetric


logger = logging.getLogger(__name__)


def default_optimization_loss(
    logits, labels, ignore_index: int = 0, background_loss_weight: float = 0.01
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

    ce_loss_fn = CrossEntropyLoss(ignore_index=ignore_index)

    ce_loss = ce_loss_fn.forward(logits, labels)

    background_dice_loss_fn = DiceLoss()

    background_dice_loss = background_dice_loss_fn.forward(logits, labels)

    background_focal_loss_fn = FocalLoss()

    background_focal_loss = background_focal_loss_fn.forward(logits, labels)

    background_ce_loss_fn = CrossEntropyLoss()

    background_ce_loss = background_ce_loss_fn.forward(logits, labels)

    loss = ce_loss + focal_loss
    background_loss = background_ce_loss + background_focal_loss

    return {
        "loss": loss + 0.1 * background_loss,
        "ce_loss": ce_loss,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "background_loss": background_loss,
        "background_dice_loss": background_dice_loss,
        "background_focal_loss": background_focal_loss,
        "background_ce_loss": background_ce_loss,
    }


def md_optimization_loss(
    logits, labels, ignore_index: int = 0, background_loss_weight: float = 0.01
):
    """
    📝 MD Optimization Loss
    Args:
        logits: (B, C, H, W)
        labels: (B, 1, H, W)
    """
    dice_loss_fn = DiceLoss(ignore_index=ignore_index)

    dice_loss = dice_loss_fn.forward(logits, labels)

    focal_loss_fn = FocalLoss(ignore_index=ignore_index)

    focal_loss = focal_loss_fn.forward(logits, labels)

    ce_loss_fn = CrossEntropyLoss(ignore_index=ignore_index)

    ce_loss = ce_loss_fn.forward(logits, labels)

    background_dice_loss_fn = DiceLoss()

    background_dice_loss = background_dice_loss_fn.forward(logits, labels)

    background_focal_loss_fn = FocalLoss()

    background_focal_loss = background_focal_loss_fn.forward(logits, labels)

    background_ce_loss_fn = CrossEntropyLoss()

    background_ce_loss = background_ce_loss_fn.forward(logits, labels)

    loss = dice_loss
    background_loss = background_dice_loss

    return {
        "loss": loss + 0.5 * background_loss,
        "ce_loss": ce_loss,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "background_loss": background_loss,
        "background_dice_loss": background_dice_loss,
        "background_focal_loss": background_focal_loss,
        "background_ce_loss": background_ce_loss,
    }


class SegmentationAdapterOptions(Enum):
    """
    📝 Segmentation Adapter Options
    """

    TRANSFORMER = "transformer"
    SIMPLE = "simple"


class SegmentationLossOptions(Enum):
    """
    📝 Segmentation Loss Options
    """

    DEFAULT = "default"
    MD = "md"


@configurable(
    group="adapter",
    name="segmentation-adapter",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
class SegmentationAdapter(nn.Module):
    def __init__(
        self,
        encoder: GATEncoder,
        num_classes: int = 100,
        background_loss_weight: float = 0.01,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 0,
        output_target_image_size: int = 256,
        decoder_target_image_size: tuple = (64, 64),
        decoder_num_blocks: int = 2,
        decoder_num_heads: int = 8,
        decoder_dropout_rate: float = 0.5,
        decoder_pre_output_dropout_rate: float = 0.3,
        decoder_layer_type: str = SegmentationAdapterOptions.TRANSFORMER.value,
        loss_type_id: str = SegmentationLossOptions.DEFAULT.value,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.class_names = (
            class_names
            if class_names
            else [str(i) for i in range(num_classes)]
        )
        self.ignore_index = ignore_index
        self.decoder_embedding_dimension = self.encoder.num_in_features_image
        self.output_target_image_size = output_target_image_size

        if loss_type_id == SegmentationLossOptions.DEFAULT.value:
            self.loss_fn = default_optimization_loss
        elif loss_type_id == SegmentationLossOptions.MD.value:
            self.loss_fn = md_optimization_loss

        decoder_layer_mapping = {
            "transformer": TransformerSegmentationDecoder(
                num_classes=num_classes,
                target_image_size=decoder_target_image_size[0],
                hidden_size=self.decoder_embedding_dimension,
                pre_output_dropout_rate=0.0,
                dropout_rate=0.0,
                decoder_num_blocks=4,
                decoder_num_heads=8,
            ),
            "simple": ChannelMixerDecoder(
                num_classes=num_classes,
                target_image_size=decoder_target_image_size[0],
                hidden_size=self.decoder_embedding_dimension,
                decoder_num_blocks=decoder_num_blocks,
                pre_output_dropout_rate=decoder_pre_output_dropout_rate,
                dropout_rate=decoder_dropout_rate,
            ),
        }

        self.decoder_head = decoder_layer_mapping[decoder_layer_type]

        self.iou_metric = IoUMetric(
            num_classes,
            ignore_index,
            {i: name for i, name in enumerate(self.class_names)},
        )

        self.background_loss_weight = background_loss_weight

    @ensemble_marker
    def compute_across_set_metrics(self):
        metrics = self.iou_metric.compute_metrics()
        self.iou_metric.pretty_print(metrics=metrics)
        self.iou_metric.reset()  # Resetting the metrics after computation
        return {k: v for k, v in metrics.items() if "per_class" not in k}

    def forward(self, image, labels: Optional[torch.Tensor] = None):
        features = self.encoder(image)["image"]["per_layer_raw_features"]
        # feature shape is either B, C, H, W or B, (W * H), C
        mask_predictions = self.decoder_head(features)

        logits = F.interpolate(
            input=mask_predictions,
            size=(
                self.output_target_image_size,
                self.output_target_image_size,
            ),
            mode="bicubic",
            align_corners=True,
        )

        output = {"logits": logits.detach()}

        if labels is not None:
            output.update(self.compute_loss_and_metrics(logits, labels))

        return output

    @ensemble_marker
    def compute_loss_and_metrics(self, logits, labels):
        loss_and_metrics = self.loss_fn(
            logits,
            labels,
            ignore_index=self.ignore_index,
            background_loss_weight=self.background_loss_weight,  # Assuming optimization_loss is defined elsewhere
        )

        if not self.training:
            preds = torch.argmax(logits, dim=1)
            labels = labels.squeeze()
            self.iou_metric.update(preds, labels)

        return loss_and_metrics

    @property
    def modality_config(self):
        return TargetModalityConfig(image=[SourceModalityConfig(image=True)])

    @property
    def encoder_transforms(self):
        return self.encoder.get_transforms()

    def init_weights(self):
        """Initialize the weights of the model."""
        # Assuming `reinit` is a function that initializes the weights
        reinit(self)

    def adapter_transforms(self, inputs: dict):
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
