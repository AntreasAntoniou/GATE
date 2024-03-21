import logging
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.config.variables import HYDRATED_IGNORE_INDEX, HYDRATED_NUM_CLASSES
from gate.metrics.segmentation import (
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    IoUMetric,
)
from gate.models.adapters import BaseAdapterModule
from gate.models.adapters.utils.helpers import reinit
from gate.models.backbones import GATEncoder
from gate.models.blocks.segmentation import TransformerSegmentationDecoder
from gate.models.core import SourceModalityConfig, TargetModalityConfig

logger = logging.getLogger(__name__)


class ImageSegmentationLoss:
    def __init__(
        self,
        ignore_index: int = 0,
        background_loss_weight: float = 0.01,
        dice_loss_weight: float = 1.0,
        focal_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
    ) -> None:
        self.ignore_index = ignore_index
        self.background_loss_weight = background_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        📝 Optimization Loss
        Args:
            logits: (B, C, H, W)
            labels: (B, 1, H, W)
        """
        dice_loss_fn = DiceLoss(ignore_index=self.ignore_index)

        dice_loss = dice_loss_fn.forward(logits, labels)

        focal_loss_fn = FocalLoss(ignore_index=self.ignore_index)

        focal_loss = focal_loss_fn.forward(logits, labels)

        ce_loss_fn = CrossEntropyLoss(ignore_index=self.ignore_index)

        ce_loss = ce_loss_fn.forward(logits, labels)

        background_dice_loss_fn = DiceLoss()

        background_dice_loss = background_dice_loss_fn.forward(logits, labels)

        background_focal_loss_fn = FocalLoss()

        background_focal_loss = background_focal_loss_fn.forward(
            logits, labels
        )

        background_ce_loss_fn = CrossEntropyLoss()

        background_ce_loss = background_ce_loss_fn.forward(logits, labels)

        loss = torch.mean(
            torch.stack(
                [
                    weight * loss_item
                    for weight, loss_item in zip(
                        [
                            self.dice_loss_weight,
                            self.focal_loss_weight,
                            self.ce_loss_weight,
                        ],
                        [dice_loss, focal_loss, ce_loss],
                    )
                    if weight > 0
                ]
            )
        )
        background_loss = torch.mean(
            torch.stack(
                [
                    weight * loss_item
                    for weight, loss_item in zip(
                        [
                            self.dice_loss_weight,
                            self.focal_loss_weight,
                            self.ce_loss_weight,
                        ],
                        [
                            background_dice_loss,
                            background_focal_loss,
                            background_ce_loss,
                        ],
                    )
                    if weight > 0
                ]
            )
        )

        return {
            "loss": loss + self.background_loss_weight * background_loss,
            "ce_loss": ce_loss,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "background_loss": background_loss,
            "background_dice_loss": background_dice_loss,
            "background_focal_loss": background_focal_loss,
            "background_ce_loss": background_ce_loss,
        }


def test_image_segmentation_loss():
    loss_fn = ImageSegmentationLoss()
    logits = torch.randn(1, 3, 256, 256, requires_grad=True)
    labels = torch.randint(0, 3, (1, 1, 256, 256))
    loss: Dict[str, torch.Tensor] = loss_fn(logits, labels)
    assert isinstance(loss, dict)
    assert "loss" in loss
    assert "ce_loss" in loss
    assert "dice_loss" in loss
    assert "focal_loss" in loss
    assert "background_loss" in loss
    assert "background_dice_loss" in loss
    assert "background_focal_loss" in loss
    assert "background_ce_loss" in loss
    assert loss["loss"] > 0
    assert loss["ce_loss"] > 0
    assert loss["dice_loss"] > 0
    assert loss["focal_loss"] > 0
    assert loss["background_loss"] > 0
    assert loss["background_dice_loss"] > 0
    assert loss["background_focal_loss"] > 0
    assert loss["background_ce_loss"] > 0
    print(loss["loss"])
    loss["loss"].backward()


class MedicalImageSegmentationLoss:
    def __init__(
        self,
        ignore_index: int = 0,
        background_loss_weight: float = 0.01,
    ) -> None:
        self.ignore_index = ignore_index
        self.background_loss_weight = background_loss_weight

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        📝 Optimization Loss
        Args:
            logits: (B, C, H, W)
            labels: (B, 1, H, W)
        """
        dice_loss_fn = DiceLoss(ignore_index=self.ignore_index)
        dice_loss = dice_loss_fn.forward(logits, labels)

        background_dice_loss_fn = DiceLoss()
        background_dice_loss = background_dice_loss_fn.forward(logits, labels)

        loss = dice_loss + self.background_loss_weight * background_dice_loss
        background_loss = background_dice_loss

        return {
            "loss": loss,
            "dice_loss": dice_loss,
            "background_loss": background_loss,
            "background_dice_loss": background_dice_loss,
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
    defaults=dict(
        num_classes=HYDRATED_NUM_CLASSES, ignore_index=HYDRATED_IGNORE_INDEX
    ),
)
class SegmentationAdapter(BaseAdapterModule):
    def __init__(
        self,
        encoder: GATEncoder,
        freeze_encoder: bool = False,
        num_classes: int = 100,
        class_names: Optional[List[str]] = None,
        output_target_image_size: int = 256,
        decoder_target_image_size: tuple = (64, 64),
        loss_type_id: str = SegmentationLossOptions.DEFAULT.value,
        ignore_index: int = 0,
        background_loss_weight: float = 0.01,
        dice_loss_weight: float = 1.0,
        focal_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
        use_batch_level_attention: bool = False,
        use_stem_instance_norm: bool = False,
    ):
        super().__init__(
            encoder=encoder,
            freeze_encoder=freeze_encoder,
            use_stem_instance_norm=use_stem_instance_norm,
        )

        self.num_classes = num_classes
        self.class_names = (
            class_names
            if class_names
            else [str(i) for i in range(num_classes)]
        )
        self.ignore_index = ignore_index
        self.decoder_embedding_dimension = self.encoder.num_in_features_image
        self.output_target_image_size = output_target_image_size
        self.use_stem_instance_norm = use_stem_instance_norm

        if loss_type_id == SegmentationLossOptions.DEFAULT.value:
            self.loss_fn = ImageSegmentationLoss(
                ignore_index=ignore_index,
                background_loss_weight=background_loss_weight,
                dice_loss_weight=dice_loss_weight,
                focal_loss_weight=focal_loss_weight,
                ce_loss_weight=ce_loss_weight,
            )
        elif loss_type_id == SegmentationLossOptions.MD.value:
            self.loss_fn = MedicalImageSegmentationLoss(
                ignore_index=ignore_index,
                background_loss_weight=background_loss_weight,
            )

        decoder_layer_mapping = TransformerSegmentationDecoder(
            num_classes=num_classes,
            target_image_size=decoder_target_image_size[0],
            hidden_size=1024,
            pre_output_dropout_rate=0.0,
            dropout_rate=0.0,
            decoder_num_blocks=8,
            decoder_num_heads=8,
        )
        self.use_batch_level_attention = use_batch_level_attention

        self.iou_metric = None
        self.iou_metric_complete = None
        self.spatial_decoder_head = decoder_layer_mapping
        self.batch_processing_head = None

        self.background_loss_weight = background_loss_weight
        self.build()
        self.freeze_encoder = freeze_encoder

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        if self.freeze_encoder:
            return self.spatial_decoder_head.parameters()

        return super().parameters(recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        if self.freeze_encoder:
            return self.spatial_decoder_head.named_parameters(prefix, recurse)

        return super().named_parameters(prefix, recurse)

    @property
    @ensemble_marker
    def iou_metrics_dict(self):
        if self.iou_metric is None:
            self.iou_metric = IoUMetric(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                class_idx_to_name={
                    i: name for i, name in enumerate(self.class_names)
                },
            )
        if self.iou_metric_complete is None:
            self.iou_metric_complete = IoUMetric(
                num_classes=self.num_classes,
                ignore_index=None,
                class_idx_to_name={
                    i: name for i, name in enumerate(self.class_names)
                },
            )
        return dict(
            iou_metric=self.iou_metric,
            iou_metric_complete=self.iou_metric_complete,
        )

    def build(self):
        dummy_batch = {
            "image": torch.randn(
                1,
                3,
                self.encoder.image_shape[0],
                self.encoder.image_shape[1],
            ),
            "labels": torch.randint(0, self.num_classes, (1, 1, 256, 256)),
        }
        _ = self(**dummy_batch)

    @ensemble_marker
    def compute_across_set_metrics(self):
        metrics = self.iou_metrics_dict["iou_metric"].compute_metrics()
        self.iou_metrics_dict["iou_metric"].pretty_print(metrics=metrics)
        self.iou_metrics_dict[
            "iou_metric"
        ].reset()  # Resetting the metrics after computation
        metrics_with_ignore = {
            k: v for k, v in metrics.items() if "per_class" not in k
        }

        complete_metrics = self.iou_metrics_dict[
            "iou_metric_complete"
        ].compute_metrics()
        self.iou_metrics_dict["iou_metric_complete"].pretty_print(
            metrics=complete_metrics
        )
        self.iou_metrics_dict["iou_metric_complete"].reset()
        metrics_complete = {
            f"{k}_complete": v
            for k, v in complete_metrics.items()
            if "per_class" not in k
        }

        return metrics_with_ignore | metrics_complete

    def forward(
        self, image: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        if self.use_stem_instance_norm:
            image = self.stem_instance_norm(image)

        features = self.encoder(image)["image"]["per_layer_raw_features"]
        # feature shape is either B, C, H, W or B, (W * H), C
        mask_predictions = self.spatial_decoder_head(features)

        # if (
        #     self.use_batch_level_attention
        #     and self.batch_processing_head is None
        # ):
        #     mask_predictions = rearrange(
        #         "b c h w -> b (h w c)", mask_predictions
        #     )
        #     mask_predictions = mask_predictions.unsqueeze(0)
        #     transformer_encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=mask_predictions.shape[-1],
        #         nhead=8,
        #         dim_feedforward=mask_predictions.shape[-1],
        #         dropout=0.0,
        #         activation="gelu",
        #         batch_first=True,
        #     )

        #     self.batch_processing_head = nn.TransformerEncoder(
        #         encoder_layer=transformer_encoder_layer,
        #         num_layers=4,
        #         norm=nn.LayerNorm(mask_predictions.shape[-1]),
        #     )

        #     self.final_conv = nn.Conv1d(
        #         mask_predictions.shape[-1], self.num_classes, kernel_size=1
        #     )

        # if self.use_batch_level_attention and self.batch_processing_head:
        #     mask_predictions = rearrange(
        #         "b c h w -> b (h w c)", mask_predictions
        #     )
        #     mask_predictions = mask_predictions.unsqueeze(0)
        #     mask_predictions = self.batch_processing_head(mask_predictions)
        #     mask_predictions = rearrange(
        #         "p b (h w c) -> (p b) c (h w)", mask_predictions
        #     )
        #     mask_predictions = self.final_conv(mask_predictions)
        #     mask_predictions = rearrange(
        #         "b c (h w) -> b c h w", mask_predictions
        #     )

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
        loss_and_metrics = self.loss_fn(logits, labels)

        if not self.training:
            preds = torch.argmax(logits, dim=1)
            labels = labels.squeeze()
            for value in self.iou_metrics_dict.values():
                value.update(preds, labels)

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
