from collections import OrderedDict
import math
from functools import partial
import time
from typing import Dict, List, Optional
from datasets.table import pa

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from timm.models.vision_transformer import Block
from transformers import (
    SamMaskDecoderConfig,
    SamModel,
    SegformerConfig,
    SegformerDecodeHead,
)
from transformers.models.sam.modeling_sam import SamFeedForward, SamMaskDecoder
from gate.boilerplate.utils import get_logger

logger = get_logger(__name__)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


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


class PositionalEncoding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.positional_encoding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] or [batch_size, channels, height, width]
        """
        is_image_based = len(x.shape) == 4
        if self.positional_encoding is None:
            if len(x.shape) == 4:
                # [batch_size, channels, height, width]
                max_len = x.shape[2] * x.shape[3]
                d_model = x.shape[1]
            else:
                max_len = x.shape[1]
                d_model = x.shape[2]

            position = torch.arange(max_len).unsqueeze(1).to(x.device)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            ).to(x.device)
            pe = torch.zeros(1, max_len, d_model).to(x.device)
            pe[0, :, 0::2] = torch.sin(position * div_term).to(x.device)
            pe[0, :, 1::2] = torch.cos(position * div_term).to(x.device)

            if is_image_based:
                pe = pe.reshape(
                    1,
                    x.shape[2],
                    x.shape[3],
                    x.shape[1],
                ).permute(0, 3, 1, 2)

            self.positional_encoding = torch.nn.Parameter(
                pe, requires_grad=True
            )

        self.positional_encoding = self.positional_encoding.to(x.device)
        x = x + self.positional_encoding[: x.shape[0]]
        return x


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
                nn.LeakyReLU(),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
                nn.LeakyReLU(),
            )
            self.pixel_wise_mlps.append(mlp)

        self.fuse_features = nn.Conv2d(
            hidden_size * self.num_feature_maps, hidden_size, kernel_size=1
        )
        self.final_conv = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(
            size=self.target_size, mode="bicubic", align_corners=True
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
                    linear_proj = nn.Linear(
                        sequence_length, target_sequence
                    )  # Apply linear projection
                    x = x.permute([0, 2, 1])  # (b, features, sequence)
                    x = x.reshape(
                        -1, sequence_length
                    )  # (b * features, sequence)
                    x = linear_proj(x).reshape(
                        -1, num_features, target_sequence
                    )

                    x = x.reshape(-1, num_features, square_root, square_root)

            # Apply pixel-wise MLP
            # logger.info(f"Input shape: {x.shape}, MLP: {mlp}")
            processed_x = mlp(x)
            # Upscale the result to the target size
            processed_x = self.upsample(processed_x)
            processed_features.append(processed_x)

        # Concatenate the processed features along the channel dimension
        fused_features = torch.cat(processed_features, dim=1)

        # Fuse the features, apply the final convolution layers, and upscale to target size
        fused_features = self.fuse_features(fused_features)
        class_features = self.final_conv(fused_features)

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
        self.positional_encoding = None

        self.decoder_embedding_dimension = decoder_embed_dim

        self.decoder = None

        self.focal_loss = FocalLoss(alpha=0.25, gamma=2, ignore_index=0)
        self.dice_loss = DiceLoss(ignore_index=0)
        self.weighted_bce = WeightedCrossEntropyLoss(
            ignore_index=0, reduction="mean"
        )
        self.debug_mode = False

    def optimization_loss(self, logits, labels):
        focal_loss = self.focal_loss(logits, labels)
        dice_loss = self.dice_loss(logits, labels)
        wce_loss = self.weighted_bce(logits, labels)

        loss = wce_loss
        return {
            "loss": loss,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "wce_loss": wce_loss,
        }

    def compute_loss_and_metrics(
        self, logits, labels: Optional[torch.Tensor] = None
    ):
        output_dict = {}
        if labels is not None:
            try:
                output_dict = self.optimization_loss(logits, labels)
            except Exception as e:
                logger.exception(f"Exception: {e}")
            if not self.training:
                try:
                    metrics = fast_miou(logits, labels)
                except Exception as e:
                    logger.info(f"Exception: {e}")
                    metrics = {}
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
            logger.info(f"Image shape: {image.shape}")
            logger.info(
                f"Mean: {image.mean()}, Std: {image.std()}, Max: {image.max()}, Min: {image.min()}"
            )

        batch, _, height, width = image.shape
        features = self.encoder(image)["image"]["per_layer_raw_features"]

        if self.decoder is None:
            self.decoder = SimpleSegmentationDecoder(
                input_feature_maps=features,
                num_classes=self.num_classes,
                target_image_size=(256, 256),
                hidden_size=self.decoder_embedding_dimension,
            )

        if self.decoder is not None:
            mask_predictions = self.decoder(features)

        output = {
            "logits": mask_predictions,
        }

        if return_loss_and_metrics:
            try:
                output |= self.compute_loss_and_metrics(
                    logits=output["logits"], labels=labels
                )

            except Exception as e:
                logger.info(f"Exception: {e}")

        return output
