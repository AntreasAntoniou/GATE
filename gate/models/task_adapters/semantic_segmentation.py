import math
import time
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.table import pa
from rich import print
from timm.models.vision_transformer import Block
from transformers import (
    SamMaskDecoderConfig,
    SamModel,
    SegformerConfig,
    SegformerDecodeHead,
)
from transformers.activations import AccurateGELUActivation
from transformers.models.sam.modeling_sam import (
    SamFeedForward,
    SamLayerNorm,
    SamMaskDecoder,
)

from gate.boilerplate.utils import get_logger

logger = get_logger(__name__)


class ResidualUpscaleConvBlock(nn.Module):
    """
    📝 Residual Convolutional Block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.activation1 = nn.GELU()
        self.norm1 = nn.InstanceNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.activation2 = nn.GELU()
        self.norm2 = nn.InstanceNorm2d(out_channels)

        self.channel_mixing = None

        self.up1 = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.activation1(out)
        out = self.norm1(out)

        out = F.interpolate(out, scale_factor=2, mode="bicubic")

        out = self.conv2(out)
        out = self.activation2(out)
        out = self.norm2(out)

        if self.up1 is None:
            self.up1 = nn.Upsample(
                size=(out.shape[2], out.shape[3]),
                mode="bicubic",
                align_corners=True,
            )
        residual = self.up1(residual)

        if out.shape[1] > residual.shape[1]:
            frame = torch.zeros_like(out)
            frame[:, : residual.shape[1], :, :] = residual
            residual = frame
        else:
            if self.channel_mixing is None:
                self.channel_mixing = nn.Conv2d(
                    residual.shape[1],
                    out.shape[1],
                    kernel_size=1,
                    stride=1,
                )
            residual = self.channel_mixing(residual)

        return out + residual


class ResidualConvBlock(nn.Module):
    """
    📝 Residual Convolutional Block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.activation1 = nn.GELU()
        self.norm1 = nn.InstanceNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.channel_mixing = None
        self.activation2 = nn.GELU()
        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.activation1(out)
        out = self.norm1(out)

        out = self.conv2(out)
        out = self.activation2(out)
        out = self.norm2(out)

        if out.shape[1] > residual.shape[1]:
            frame = torch.zeros_like(out)
            frame[:, : residual.shape[1], :, :] = residual
            residual = frame
        else:
            if self.channel_mixing is None:
                self.channel_mixing = nn.Conv2d(
                    residual.shape[1],
                    out.shape[1],
                    kernel_size=1,
                    stride=1,
                )
            residual = self.channel_mixing(residual)

        return out + residual


from gate.metrics.segmentation import (
    DiceLoss,
    FocalLoss,
    WeightedCrossEntropyLoss,
    diff_dice_loss,
    diff_sigmoid_focal_loss,
    fast_miou,
)


def optimization_loss(logits, labels):
    """
    📝 Optimization Loss
    Args:
        logits: (B, C, H, W)
        labels: (B, 1, H, W)
    """
    # print(f"logits.shape: {logits.shape}, labels.shape: {labels.shape}")
    logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = labels.reshape(-1)

    non_background_indices = labels != 0
    non_background_logits = logits[non_background_indices]
    non_background_labels = labels[non_background_indices]

    background_indices = labels == 0
    background_logits = logits[background_indices]
    background_labels = labels[background_indices]

    # get number of unique classes

    cross_entropy_loss = F.cross_entropy(
        non_background_logits, non_background_labels
    )
    dice_loss = diff_dice_loss(non_background_logits, non_background_labels)
    focal_loss = diff_sigmoid_focal_loss(
        non_background_logits, non_background_labels, alpha=0.25, gamma=2.0
    )

    background_cross_entropy_loss = F.cross_entropy(
        background_logits, background_labels
    )
    background_dice_loss = diff_dice_loss(background_logits, background_labels)
    background_focal_loss = diff_sigmoid_focal_loss(
        background_logits,
        background_labels,
        alpha=0.25,
        gamma=2.0,
    )
    background_loss = (
        background_cross_entropy_loss
        # background_dice_loss
        # + background_focal_loss
    )

    loss = (
        cross_entropy_loss
        # dice_loss
        # + focal_loss
        + 0.1 * background_loss
    )

    return {
        "loss": loss,
        "cross_entropy_loss": cross_entropy_loss,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "background_loss": background_loss,
        "background_cross_entropy_loss": background_cross_entropy_loss,
        "background_dice_loss": background_dice_loss,
        "background_focal_loss": background_focal_loss,
    }


class UpscaleMultiBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        num_blocks: int = 2,
        encoder_features: int = 64,
    ):
        super().__init__()
        self.in_channel_mixing = None
        self.channel_mixing = None

        self.in_features = in_features
        self.encoder_features = encoder_features
        self.upscale_net = ResidualUpscaleConvBlock(
            in_channels=in_features + encoder_features,
            out_channels=hidden_size,
        )

        self.detail_conv = nn.Sequential(
            OrderedDict(
                {
                    f"upscale_block_{idx}": ResidualConvBlock(
                        hidden_size, hidden_size
                    )
                    for idx in range(num_blocks)
                }
            )
        )

        self.out_conv = nn.Conv2d(
            hidden_size, out_features, kernel_size=1, stride=1
        )

    def forward(
        self, x: torch.Tensor, encoder_features: torch.Tensor
    ) -> torch.Tensor:
        if self.channel_mixing is None:
            self.channel_mixing = nn.Conv2d(
                encoder_features.shape[1],
                self.encoder_features,
                kernel_size=1,
                stride=1,
            )

        if self.in_channel_mixing is None and x.shape[1] != self.in_features:
            self.in_channel_mixing = nn.Conv2d(
                x.shape[1], self.in_features, kernel_size=1, stride=1
            )

        if self.in_channel_mixing is not None:
            x = self.in_channel_mixing(x)

        encoder_features = self.channel_mixing(encoder_features)

        if (
            x.shape[-2] != encoder_features.shape[-2]
            or x.shape[-1] != encoder_features.shape[-1]
        ):
            encoder_features = F.interpolate(
                encoder_features,
                size=(x.shape[-2], x.shape[-1]),
                mode="bicubic",
                align_corners=True,
            )

        cat_features = torch.cat([x, encoder_features], dim=1)

        out = self.upscale_net(cat_features)

        out = self.detail_conv(out)

        out = self.out_conv(out)

        return out


import math


def has_exact_square_root(s: int) -> bool:
    # Get the size of the second dimension (s)

    # Calculate the square root of s
    root = math.sqrt(s)

    # Check if the square root is an integer
    return root.is_integer()


class SimpleSegmentationDecoder(nn.Module):
    def __init__(
        self,
        input_feature_maps: List[torch.Tensor],
        num_classes: int,
        target_image_size: tuple,
        hidden_size: int = 256,
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
        self.target_size = target_image_size

        self.pixel_wise_mlps = nn.ModuleList()

        for input_features in input_feature_maps:
            if len(input_features.shape) == 4:
                in_channels = input_features.shape[1]
            elif len(input_features.shape) == 3:
                in_channels = input_features.shape[2]

            mlp = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size, kernel_size=1),
                SamLayerNorm(
                    normalized_shape=hidden_size, data_format="channels_first"
                ),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
                SamLayerNorm(
                    normalized_shape=hidden_size, data_format="channels_first"
                ),
                nn.LeakyReLU(inplace=True),
            )
            self.pixel_wise_mlps.append(mlp)

        self.fuse_features = nn.Conv2d(
            hidden_size * self.num_feature_maps, hidden_size * 2, kernel_size=1
        )
        self.fuse_features_norm = nn.LazyInstanceNorm2d()

        self.fuse_features_act = AccurateGELUActivation()
        self.final_conv = nn.Conv2d(
            hidden_size * 2, num_classes, kernel_size=1
        )
        self.upsample = nn.Upsample(
            size=self.target_size, mode="bilinear", align_corners=True
        )

    def forward(self, input_list: list):
        """
        Forward pass of SimpleSegmentationDecoder.

        📮 Passes the input list through pixel-wise MLPs, fuses the features, and upscales the result to the target size.

        :param input_list: List of input tensors, either shape (b, c, h, w) or (b, sequence, features).
        :return: Output tensor representing class feature maps of shape (b, num_classes, target_h, target_w).
        """
        processed_features = []
        for mlp, x in zip(self.pixel_wise_mlps, input_list):
            # Check if input is (b, sequence, features)
            start_time = time.time()
            if len(x.shape) == 3:
                sequence_length = x.shape[1]
                num_features = x.shape[2]
                square_root = int(math.sqrt(sequence_length))

                # If the sequence has an exact square root, reshape it to (b, c, h, w) format
                if sequence_length == square_root * square_root:
                    c = num_features
                    h = w = square_root
                    x = x.permute([0, 2, 1]).reshape(-1, c, h, w)
                # If not, apply a linear projection
                else:
                    target_sequence = square_root**2
                    x = x.permute([0, 2, 1])  # (b, features, sequence)
                    x = F.adaptive_avg_pool1d(x, target_sequence)

                    x = x.reshape(-1, num_features, square_root, square_root)
            logger.debug(f"Reshaping took {time.time() - start_time} seconds")
            # Apply pixel-wise MLP
            # logger.debug(f"Input shape: {x.shape}, MLP: {mlp}")
            start_time = time.time()
            processed_x = mlp(x)
            logger.debug(f"MLP took {time.time() - start_time} seconds")
            # Upscale the result to the target size
            start_time = time.time()
            processed_x = self.upsample(processed_x)
            logger.debug(f"Upsampled shape: {processed_x.shape}")
            logger.debug(f"Upsampling took {time.time() - start_time} seconds")
            processed_features.append(processed_x)

        # Concatenate the processed features along the channel dimension
        start_time = time.time()
        fused_features = torch.cat(processed_features, dim=1)
        logger.debug(f"Concatenation took {time.time() - start_time} seconds")

        # Fuse the features, apply the final convolution layers, and upscale to target size
        start_time = time.time()
        fused_features = self.fuse_features(fused_features)
        fused_norm_features = self.fuse_features_norm(fused_features)
        fused_act_features = self.fuse_features_act(fused_norm_features)
        logger.debug(
            f"Fusing features took {time.time() - start_time} seconds"
        )
        start_time = time.time()
        class_features = self.final_conv(fused_act_features)
        logger.debug(
            f"Final convolution took {time.time() - start_time} seconds"
        )

        return class_features


class PreResizeSimpleSegmentationDecoder(nn.Module):
    def __init__(
        self,
        input_feature_maps: List[torch.Tensor],
        num_classes: int,
        target_image_size: tuple,
        hidden_size: int = 256,
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
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(
                    self.num_blocks * hidden_size,
                    self.num_blocks * hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm2d(
                    num_features=self.num_blocks * hidden_size,
                ),
                nn.LeakyReLU(inplace=True),
            )

            self.fuse_features = nn.Conv2d(
                self.num_blocks * hidden_size, hidden_size, kernel_size=1
            )
            self.fuse_features_norm = nn.LazyInstanceNorm2d()

            self.fuse_features_act = AccurateGELUActivation()
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
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(
                    self.num_blocks * hidden_size,
                    self.num_blocks * hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm1d(num_features=self.num_blocks * hidden_size),
                nn.LeakyReLU(inplace=True),
            )

            self.fuse_features = nn.Conv1d(
                self.num_blocks * hidden_size, hidden_size, kernel_size=1
            )
            self.fuse_features_norm = nn.LazyInstanceNorm1d()

            self.fuse_features_act = AccurateGELUActivation()
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


class SegmentationViT(nn.Module):
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

        self.decoder = None

        self.focal_loss = FocalLoss(
            alpha=0.25, gamma=2, ignore_index=self.background_class
        )
        self.dice_loss = DiceLoss(ignore_index=self.background_class)
        self.background_focal_loss = FocalLoss(alpha=0.25, gamma=2)
        self.background_dice_loss = DiceLoss()

        self.debug_mode = False

    def optimization_loss(self, logits, labels):
        focal_loss = self.focal_loss(logits, labels)
        dice_loss = self.dice_loss(logits, labels)
        background_focal_loss = self.background_focal_loss(logits, labels)
        background_dice_loss = self.background_dice_loss(logits, labels)

        loss = (
            focal_loss
            + dice_loss
            + self.background_weight
            * (background_focal_loss + background_dice_loss)
        )
        return {
            "loss": loss,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "background_focal_loss": background_focal_loss,
            "background_dice_loss": background_dice_loss,
        }

    def compute_loss_and_metrics(
        self, logits, labels: Optional[torch.Tensor] = None
    ):
        output_dict = {}
        if labels is not None:
            try:
                output_dict = self.optimization_loss(logits, labels)
            except Exception as e:
                raise e
            if not self.training:
                try:
                    metrics = fast_miou(
                        logits, labels, self.ignore_index, self.class_names
                    )
                except Exception as e:
                    raise e
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

        batch, _, height, width = image.shape
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

        if self.decoder is None:
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

            self.decoder = PreResizeSimpleSegmentationDecoder(
                input_feature_maps=features,
                num_classes=self.num_classes,
                target_image_size=target_image_size,
                hidden_size=self.decoder_embedding_dimension,
            )

        start_time = time.time()
        if self.decoder is not None:
            mask_predictions = self.decoder(features)

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
            try:
                output |= self.compute_loss_and_metrics(
                    logits=logits, labels=labels
                )

            except Exception as e:
                logger.debug(f"Exception: {e}")

        return output
