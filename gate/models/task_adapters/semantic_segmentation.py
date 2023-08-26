import math
import time
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.table import pa
from mmseg.evaluation.metrics import IoUMetric
from rich import print

from gate.boilerplate.utils import get_logger
from gate.metrics.segmentation import (
    DiceLoss,
    FocalLoss,
    WeightedCrossEntropyLoss,
    miou_metrics,
)

logger = get_logger(__name__)


def has_exact_square_root(s: int) -> bool:
    # Get the size of the second dimension (s)

    # Calculate the square root of s
    root = math.sqrt(s)

    # Check if the square root is an integer
    return root.is_integer()


def optimization_loss(logits, labels, ignore_index: int = 0):
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
        "loss": loss + 0.05 * background_loss,
        "ce_loss": ce_loss,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "background_loss": background_loss,
        "background_dice_loss": background_dice_loss,
        "background_focal_loss": background_focal_loss,
        "background_ce_loss": background_ce_loss,
    }


class PreResizeSimpleSegmentationDecoder(nn.Module):
    def __init__(
        self,
        input_feature_maps: List[torch.Tensor],
        num_classes: int,
        target_image_size: tuple,
        hidden_size: int = 256,
        dropout_rate: float = 0.5,
        pre_output_dropout_rate: float = 0.3,
    ):
        """
        SimpleSegmentationDecoder class for segmentation tasks.

        :param input_feature_maps: List of integers representing the number of feature maps of each input tensor.
        :param num_classes: Integer representing the number of classes to predict for segmentation.
        :param target_size: Tuple containing the height and width for the target output size.
        :param hidden_size: Integer representing the hidden size for pixel-wise MLP layers, default=64.
        """
        super().__init__()
        self.num_feature_maps = len(input_feature_maps)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.target_image_size = target_image_size

        self.pixel_wise_mlps = nn.ModuleList()
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bilinear", align_corners=True
        )
        self.spatial_mixer = None
        self.closest_square_root = None
        self.num_blocks = len(input_feature_maps)

        if len(input_feature_maps[0].shape) == 4:
            input_feature_maps = [
                self.upsample(x) if x.shape[-1] != target_image_size[0] else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        elif len(input_feature_maps[0].shape) == 3:
            input_feature_maps = [
                x.permute([0, 2, 1]) for x in input_feature_maps
            ]  # (b, sequence, features) -> (b, features, sequence)
            input_feature_maps = [
                F.adaptive_avg_pool1d(x, output_size=target_image_size)
                if x.shape[2] != target_image_size
                else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        if len(input_feature_maps.shape) == 4:
            in_channels = input_feature_maps.shape[1]
            self.mlp = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.num_blocks * hidden_size, kernel_size=1
                ),
                nn.InstanceNorm2d(
                    num_features=self.num_blocks * hidden_size,
                ),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=dropout_rate, inplace=False),
                nn.Conv2d(
                    self.num_blocks * hidden_size,
                    self.num_blocks * hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm2d(
                    num_features=self.num_blocks * hidden_size,
                ),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=dropout_rate, inplace=False),
            )

            self.fuse_features = nn.Conv2d(
                self.num_blocks * hidden_size, hidden_size, kernel_size=1
            )
            self.fuse_features_norm = nn.LazyInstanceNorm2d()

            self.fuse_features_act = nn.LeakyReLU()
            self.fuse_features_dropout = nn.Dropout2d(
                p=pre_output_dropout_rate, inplace=False
            )
            self.final_conv = nn.Conv2d(
                hidden_size, num_classes, kernel_size=1
            )
        elif len(input_feature_maps.shape) == 3:
            in_channels = input_feature_maps.shape[1]
            self.mlp = nn.Sequential(
                nn.Conv1d(
                    in_channels, self.num_blocks * hidden_size, kernel_size=1
                ),
                nn.InstanceNorm1d(num_features=self.num_blocks * hidden_size),
                nn.LeakyReLU(inplace=False),
                nn.Dropout1d(p=dropout_rate, inplace=False),
                nn.Conv1d(
                    self.num_blocks * hidden_size,
                    self.num_blocks * hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm1d(num_features=self.num_blocks * hidden_size),
                nn.LeakyReLU(inplace=False),
                nn.Dropout1d(p=dropout_rate, inplace=False),
            )

            self.fuse_features = nn.Conv1d(
                self.num_blocks * hidden_size, hidden_size, kernel_size=1
            )
            self.fuse_features_norm = nn.LazyInstanceNorm1d()

            self.fuse_features_act = nn.LeakyReLU()
            self.fuse_features_dropout = nn.Dropout1d(
                p=pre_output_dropout_rate, inplace=False
            )
            self.final_conv = nn.Conv1d(
                hidden_size, num_classes, kernel_size=1
            )

            sequence_length = input_feature_maps.shape[2]
            if not has_exact_square_root(sequence_length):
                self.closest_square_root = int(
                    np.floor(np.sqrt(sequence_length))
                )
                self.spatial_mixer = nn.Conv1d(
                    in_channels=sequence_length,
                    out_channels=self.closest_square_root**2,
                    kernel_size=1,
                )
            else:
                self.closest_square_root = int(np.sqrt(sequence_length))

    def forward(self, input_list: list):
        """
        Forward pass of SimpleSegmentationDecoder.

        📮 Passes the input list through pixel-wise MLPs, fuses the features, and upscales the result to the target size.

        :param input_list: List of input tensors, either shape (b, c, h, w) or (b, sequence, features).
        :return: Output tensor representing class feature maps of shape (b, num_classes, target_h, target_w).
        """
        start_time = time.time()
        input_feature_maps = input_list
        if len(input_feature_maps[0].shape) == 4:
            input_feature_maps = [
                self.upsample(x)
                if x.shape[-1] != self.target_image_size
                else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        elif len(input_feature_maps[0].shape) == 3:
            input_feature_maps = [
                x.permute([0, 2, 1]) for x in input_feature_maps
            ]  # (b, sequence, features) -> (b, features, sequence)
            input_feature_maps = [
                F.adaptive_avg_pool1d(x, output_size=self.target_image_size)
                if x.shape[2] != self.target_image_size
                else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        logger.debug(f"Upsampling took {time.time() - start_time} seconds")
        logger.debug(
            f"Shape of input feature maps: {input_feature_maps.shape}"
        )
        start_time = time.time()
        logger.debug(f"MLP summary: {self.mlp}")
        processed_features = self.mlp(input_feature_maps)
        logger.debug(f"MLP took {time.time() - start_time} seconds")
        # Concatenate the processed features along the channel dimension
        start_time = time.time()
        fused_features = processed_features
        logger.debug(f"Concatenation took {time.time() - start_time} seconds")

        # Fuse the features, apply the final convolution layers, and upscale to target size
        start_time = time.time()
        logger.debug(f"Shape of fused features: {fused_features.shape}")
        fused_features = self.fuse_features(fused_features)
        fused_norm_features = self.fuse_features_norm(fused_features)
        fused_act_features = self.fuse_features_act(fused_norm_features)
        logger.debug(
            f"Fusing features took {time.time() - start_time} seconds"
        )
        start_time = time.time()
        fused_act_features = self.fuse_features_dropout(fused_act_features)
        class_features = self.final_conv(fused_act_features)
        logger.debug(
            f"Final convolution took {time.time() - start_time} seconds"
        )

        if self.spatial_mixer is not None:
            class_features = class_features.permute([0, 2, 1])
            class_features = self.spatial_mixer(class_features)
            class_features = class_features.permute([0, 2, 1])

        if self.closest_square_root is not None:
            class_features = class_features.reshape(
                class_features.shape[0],
                self.num_classes,
                self.closest_square_root,
                self.closest_square_root,
            )

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
        input_feature_maps: List[torch.Tensor],
        num_classes: int,
        target_image_size: tuple,
        hidden_size: int = 256,
        dropout_rate: float = 0.5,
        pre_output_dropout_rate: float = 0.3,
        num_transformer_blocks: int = 4,
    ):
        """
        TransformerSegmentationDecoderHead class for segmentation tasks.

        :param input_feature_maps: List of integers representing the number of feature maps of each input tensor.
        :param num_classes: Integer representing the number of classes to predict for segmentation.
        :param target_size: Tuple containing the height and width for the target output size.
        :param hidden_size: Integer representing the hidden size for pixel-wise MLP layers, default=64.
        """
        super().__init__()
        self.num_feature_maps = len(input_feature_maps)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.target_image_size = target_image_size

        self.pixel_wise_mlps = nn.ModuleList()
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bilinear", align_corners=True
        )
        self.spatial_mixer = None
        self.closest_square_root = None
        self.num_blocks = len(input_feature_maps)

        if len(input_feature_maps[0].shape) == 4:
            input_feature_maps = [
                self.upsample(x) if x.shape[-1] != target_image_size[0] else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        elif len(input_feature_maps[0].shape) == 3:
            input_feature_maps = [
                x.permute([0, 2, 1]) for x in input_feature_maps
            ]  # (b, sequence, features) -> (b, features, sequence)
            input_feature_maps = [
                F.adaptive_avg_pool1d(x, output_size=target_image_size)
                if x.shape[2] != target_image_size
                else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        if len(input_feature_maps.shape) == 4:
            in_channels = input_feature_maps.shape[1]
            # input shape (b, c, h, w)
            self.projection_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.num_blocks * hidden_size,
                kernel_size=1,
            )

            self.fuse_features = nn.Conv1d(
                self.num_blocks * hidden_size, hidden_size, kernel_size=1
            )
            self.fuse_features_norm = nn.LazyInstanceNorm1d()

            self.fuse_features_act = nn.LeakyReLU()
            self.fuse_features_dropout = nn.Dropout1d(
                p=pre_output_dropout_rate, inplace=False
            )

            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
            )

            self.segmentation_processing_head = nn.TransformerEncoder(
                encoder_layer=transformer_encoder_layer,
                num_layers=num_transformer_blocks,
                norm=nn.LayerNorm(hidden_size),
            )

            self.final_conv = nn.Conv1d(
                hidden_size, num_classes, kernel_size=1
            )

        elif len(input_feature_maps.shape) == 3:
            in_channels = input_feature_maps.shape[1]

            self.projection_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.num_blocks * hidden_size,
                kernel_size=1,
            )

            self.fuse_features = nn.Conv1d(
                self.num_blocks * hidden_size, hidden_size, kernel_size=1
            )
            self.fuse_features_norm = nn.LazyInstanceNorm1d()

            self.fuse_features_act = nn.LeakyReLU()
            self.fuse_features_dropout = nn.Dropout1d(
                p=pre_output_dropout_rate, inplace=False
            )

            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
            )

            self.segmentation_processing_head = nn.TransformerEncoder(
                encoder_layer=transformer_encoder_layer,
                num_layers=num_transformer_blocks,
                norm=nn.LayerNorm(hidden_size),
            )

            self.final_conv = nn.Conv1d(
                hidden_size, num_classes, kernel_size=1
            )

    def forward(self, input_list: list):
        """
        Forward pass of SimpleSegmentationDecoder.

        📮 Passes the input list through pixel-wise MLPs, fuses the features, and upscales the result to the target size.

        :param input_list: List of input tensors, either shape (b, c, h, w) or (b, sequence, features).
        :return: Output tensor representing class feature maps of shape (b, num_classes, target_h, target_w).
        """
        input_feature_maps = input_list
        if len(input_feature_maps[0].shape) == 4:
            input_feature_maps = [
                self.upsample(x)
                if x.shape[-1] != self.target_image_size
                else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        elif len(input_feature_maps[0].shape) == 3:
            input_feature_maps = [
                x.permute([0, 2, 1]) for x in input_feature_maps
            ]  # (b, sequence, features) -> (b, features, sequence)
            input_feature_maps = [
                F.adaptive_avg_pool1d(x, output_size=self.target_image_size)
                if x.shape[2] != self.target_image_size
                else x
                for x in input_feature_maps
            ]
            input_feature_maps = torch.cat(input_feature_maps, dim=1)

        projected_features = self.projection_layer(input_feature_maps)

        if len(projected_features.shape) == 4:
            projected_features = projected_features.permute(
                [0, 2, 3, 1]
            ).reshape(
                projected_features.shape[0],
                projected_features.shape[2] * projected_features.shape[3],
                projected_features.shape[1],
            )
        elif len(projected_features.shape) == 3:
            projected_features = projected_features.permute([0, 2, 1]).reshape(
                projected_features.shape[0],
                projected_features.shape[2],
                projected_features.shape[1],
            )

        fused_features = projected_features
        # Fuse the features, apply the final convolution layers, and upscale to target size
        fused_features = fused_features.permute([0, 2, 1])
        fused_features = self.fuse_features(fused_features)
        fused_norm_features = self.fuse_features_norm(fused_features)
        fused_act_features = self.fuse_features_act(fused_norm_features)
        fused_act_features = self.fuse_features_dropout(fused_act_features)

        transformed_features = fused_act_features.permute([0, 2, 1])
        transformed_features = self.segmentation_processing_head(
            transformed_features
        )
        transformed_features = transformed_features.permute([0, 2, 1])

        class_features = self.final_conv(fused_act_features)

        if self.spatial_mixer is not None:
            class_features = class_features.permute([0, 2, 1])
            class_features = self.spatial_mixer(class_features)
            class_features = class_features.permute([0, 2, 1])

        class_features = upsample_tensor(class_features)

        return class_features


class SimpleSegmentationDecoder(nn.Module):
    """
    Vision Transformer for Semantic Segmentation.
    """

    def __init__(
        self,
        encoder_model: nn.Module,
        decoder_embed_dim: int = 512,
        num_classes: int = 100,
        num_patches: int = 14,
        background_weight: float = 0.01,
        background_class: int = 0,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 0,
        decoder_layer_type: Union[
            PreResizeSimpleSegmentationDecoder,
            TransformerSegmentationDecoderHead,
            str,
        ] = PreResizeSimpleSegmentationDecoder,
        **kwargs,
    ):
        """
        Construct a Vision Transformer for Semantic Segmentation.

        Args:
            encoder_model (nn.Module): Pretrained model to be used for encoding.
            embed_dim (int, optional): Embedding dimension for the transformer. Defaults to 768.
            decoder_embed_dim (int, optional): Embedding dimension for the decoder. Defaults to 768.
            decoder_depth (int, optional): Number of layers in the decoder. Defaults to 2.
            decoder_num_heads (int, optional): Number of heads in the decoder's multi-head attention mechanism. Defaults to 8.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            norm_layer (torch.nn.Module, optional): Normalization layer to use. Defaults to nn.LayerNorm.
            num_classes (int, optional): Number of classes. Defaults to 100.
        """
        super().__init__()

        self.encoder = encoder_model
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.background_weight = background_weight
        self.background_class = background_class
        self.class_names = class_names
        self.ignore_index = ignore_index

        self.decoder_embedding_dimension = decoder_embed_dim

        if decoder_layer_type == "transformer":
            self.decoder_layer_type = TransformerSegmentationDecoderHead
        elif decoder_layer_type == "simple":
            self.decoder_layer_type = PreResizeSimpleSegmentationDecoder
        else:
            self.decoder_layer_type = decoder_layer_type

        self.decoder_head = None

        self.debug_mode = True

        self.iou_metric = IoUMetric(ignore_index=self.ignore_index)

    def compute_across_set_iou(self):
        # Call the compute_metrics method
        more_metrics = self.iou_metric.compute_metrics(self.iou_metric.results)

        # metrics: {'aAcc': 65.36, 'mIoU': 6.65, 'mAcc': 9.72}
        more_metrics["overall_accuracy_mmseg"] = torch.tensor(
            more_metrics["aAcc"]
        )
        more_metrics["mean_iou_mmseg"] = torch.tensor(more_metrics["mIoU"])
        more_metrics["mean_accuracy_mmseg"] = torch.tensor(
            more_metrics["mAcc"]
        )
        self.iou_metric.results = []

        return more_metrics

    def optimization_loss(self, logits, labels):
        return optimization_loss(
            logits, labels, ignore_index=self.ignore_index
        )

    def compute_loss_and_metrics(
        self, logits, labels: Optional[torch.Tensor] = None
    ):
        output_dict = {}
        if labels is not None:
            logger.debug(
                f"Labels shape: {labels.shape}, Logits shape: {logits.shape}"
            )

            if len(labels.shape) == 4:
                labels = labels.reshape(-1, 256, 256)
            elif len(labels.shape) == 3:
                labels = labels.reshape(-1, 256, 256)

            logger.debug(
                f"Labels shape: {labels.shape}, Logits shape: {logits.shape}"
            )

            output_dict = self.optimization_loss(logits, labels)
            if not self.training:
                self.iou_metric.dataset_meta = {"classes": self.class_names}

                metrics = miou_metrics(
                    logits,
                    labels,
                    self.iou_metric,
                    self.ignore_index,
                    self.class_names,
                )
                output_dict = output_dict | metrics
        return output_dict

    def forward(
        self,
        image,
        labels: Optional[torch.Tensor] = None,
        return_loss_and_metrics: bool = True,
    ):
        """
            Forward pass for the segmentation model.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Segmentation map.
        """

        if self.debug_mode:
            logger.debug(f"Image shape: {image.shape}")
            logger.debug(
                f"Mean: {image.mean()}, Std: {image.std()}, Max: {image.max()}, Min: {image.min()}"
            )

        if len(image.shape) == 4:
            batch, channels, height, width = image.shape
        elif len(image.shape) == 5:
            batch, slices, channels, height, width = image.shape
            image = image.reshape(-1, channels, height, width)
        elif len(image.shape) == 6:
            batch, scan, slices, channels, height, width = image.shape
            image = image.reshape(-1, channels, height, width)

        start_time = time.time()
        features = self.encoder(image)["image"]["per_layer_raw_features"]

        if len(features[0].shape) == 3:
            total_feature_blocks = len(features)
            one_quarter_block = total_feature_blocks // 4
            three_quarter_block = total_feature_blocks - one_quarter_block
            features = [
                features[0],
                features[one_quarter_block],
                features[three_quarter_block],
                features[-1],
            ]

        # for f in features:
        #     logger.debug(f"Feature shape: {f.shape}")

        if self.debug_mode:
            logger.debug(f"Encoder took {time.time() - start_time} seconds")

        if self.decoder_head is None:
            feature_shapes = [x.shape for x in features]
            logger.debug(f"Feature shapes: {feature_shapes}")
            if len(features[0].shape) == 3:
                sequence_lengths = [x.shape[1] for x in features]
                largest_feature_map = max(sequence_lengths)
                max_height = largest_feature_map
                max_width = largest_feature_map
                target_image_size = largest_feature_map
            elif len(features[0].shape) == 4:
                heights = [x.shape[2] for x in features]
                max_height = max(heights)
                widths = [x.shape[3] for x in features]
                max_width = max(widths)
                target_image_size = (64, 64)
            else:
                raise ValueError(
                    f"Unsupported feature map shape: {features[0].shape}"
                )

            self.decoder_head = self.decoder_layer_type(
                input_feature_maps=features,
                num_classes=self.num_classes,
                target_image_size=target_image_size,
                hidden_size=self.decoder_embedding_dimension,
            )

        start_time = time.time()
        if self.decoder_head is not None:
            # Check the batch size
            batch_size = features.shape[0]

            # Initialize list to collect output chunks
            output_chunks = []

            # Process in chunks
            for start_idx in range(0, batch_size, 20):
                end_idx = min(start_idx + 20, batch_size)
                chunk = features[start_idx:end_idx]

                # Forward pass for this chunk
                output_chunk = self.decoder_head(chunk)

                # Collect the output
                output_chunks.append(output_chunk)

            # Concatenate all chunks back to a single tensor
            mask_predictions = torch.cat(output_chunks, dim=0)

        if self.debug_mode:
            logger.debug(f"Decoder took {time.time() - start_time} seconds")
            logger.debug(f"Mask predictions shape: {mask_predictions.shape}")

        logits = F.interpolate(
            mask_predictions,
            size=(256, 256),
            mode="bicubic",
            align_corners=True,
        )
        output = {"logits": logits.detach()}

        if return_loss_and_metrics:
            output |= self.compute_loss_and_metrics(
                logits=logits, labels=labels
            )

        return output
