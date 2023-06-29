import math
import time
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import AccurateGELUActivation
from transformers.models.sam.modeling_sam import SamLayerNorm

from gate.boilerplate.utils import get_logger

logger = get_logger(name=__name__)


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
