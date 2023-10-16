import math
import time
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.evaluation.metrics import IoUMetric as mmsegIoUMetric

from gate.boilerplate.utils import get_logger
from gate.metrics.segmentation import (
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    IoUMetric,
)
from gate.models.blocks.segmentation import (
    ChannelMixerDecoder,
    TransformerSegmentationDecoder,
)

logger = get_logger(__name__)


def optimization_loss(
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

    background_dice_loss_fn = DiceLoss(ignore_index=-1)

    background_dice_loss = background_dice_loss_fn.forward(logits, labels)

    background_focal_loss_fn = FocalLoss(ignore_index=-1)

    background_focal_loss = background_focal_loss_fn.forward(logits, labels)

    background_ce_loss_fn = CrossEntropyLoss(ignore_index=-1)

    background_ce_loss = background_ce_loss_fn.forward(logits, labels)

    loss = ce_loss
    background_loss = background_ce_loss

    return {
        "loss": loss + 0.01 * background_loss,
        "ce_loss": ce_loss,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "background_loss": background_loss,
        "background_dice_loss": background_dice_loss,
        "background_focal_loss": background_focal_loss,
        "background_ce_loss": background_ce_loss,
    }


class SegmentationAdapter(nn.Module):
    def __init__(
        self,
        encoder_model: nn.Module,
        num_classes: int = 100,
        background_loss_weight: float = 0.01,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 0,
        output_target_image_size: int = 256,
        decoder_target_image_size: tuple = (64, 64),
        decoder_embed_dim: int = 512,
        decoder_num_blocks: int = 2,
        decoder_num_heads: int = 8,
        decoder_dropout_rate: float = 0.5,
        decoder_pre_output_dropout_rate: float = 0.3,
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
        self.output_target_image_size = output_target_image_size

        # Assuming decoder_layer_mapping and other related classes and functions are defined elsewhere in the code

        decoder_layer_mapping = {
            "transformer": TransformerSegmentationDecoder(
                num_classes=num_classes,
                target_image_size=64,
                hidden_size=256,
                pre_output_dropout_rate=0.3,
                dropout_rate=0.2,
                decoder_num_blocks=4,
                decoder_num_heads=8,
            ),
            "simple": ChannelMixerDecoder(
                num_classes=num_classes,
                target_image_size=decoder_target_image_size[0],
                hidden_size=decoder_embed_dim,
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
        self.iou_metric_mmseg = mmsegIoUMetric(
            num_classes=num_classes, ignore_index=ignore_index
        )
        self.iou_metric_mmseg.dataset_meta = {
            "classes": list(range(num_classes))
        }

        self.background_loss_weight = background_loss_weight

    def compute_across_set_iou(self):
        metrics = self.iou_metric.compute_metrics()
        self.iou_metric.pretty_print(metrics=metrics)
        self.iou_metric.reset()  # Resetting the metrics after computation
        metrics = {k: v for k, v in metrics.items() if "per_class" not in k}

        mmseg_metrics = self.iou_metric_mmseg.compute_metrics(
            self.iou_metric_mmseg.results
        )
        self.iou_metric_mmseg.results.clear()
        mmseg_metrics = {f"mmseg{k}": v for k, v in mmseg_metrics.items()}
        return metrics | mmseg_metrics

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

    def compute_loss_and_metrics(self, logits, labels):
        loss_and_metrics = optimization_loss(
            logits,
            labels,
            ignore_index=self.ignore_index,
            background_loss_weight=self.background_loss_weight,  # Assuming optimization_loss is defined elsewhere
        )

        # print("logit shape: ", logits.shape)

        if not self.training:
            preds = torch.argmax(logits, dim=1)
            labels = labels.squeeze()
            self.iou_metric.update(preds, labels)

            self.iou_metric_mmseg.process(
                data_batch=None,
                data_samples=[
                    {
                        "pred_sem_seg": {"data": preds},
                        "gt_sem_seg": {"data": labels},
                    }
                ],
            )

        return loss_and_metrics
