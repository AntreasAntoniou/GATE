import math
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def has_exact_square_root(s: int) -> bool:
    # Get the size of the second dimension (s)

    # Calculate the square root of s
    root = math.sqrt(s)

    # Check if the square root is an integer
    return root.is_integer()


def upsample_tensor(input_tensor):
    b, c, s = input_tensor.shape
    new_size = math.ceil(math.sqrt(s)) ** 2
    sq_root = int(math.sqrt(new_size))
    output_tensor = F.upsample(
        input_tensor,
        size=(sq_root * sq_root),
    )
    return rearrange(
        output_tensor, "b c (h w) -> b c h w", h=sq_root, w=sq_root
    )


import logging

logger = logging.getLogger(__name__)


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


class ChannelMixerDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        target_image_size: int = 64,
        hidden_size: int = 256,
        decoder_num_blocks: int = 2,
        pre_output_dropout_rate: float = 0.3,
        dropout_rate: float = 0.5,
    ):
        """
        SimpleSegmentationDecoder class for segmentation tasks.

        :param input_feature_maps: List of integers representing the number of feature maps of each input tensor.
        :param num_classes: Integer representing the number of classes to predict for segmentation.
        :param target_size: Tuple containing the height and width for the target output size.
        :param hidden_size: Integer representing the hidden size for pixel-wise MLP layers, default=64.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.target_image_size = target_image_size
        self.dropout_rate = dropout_rate
        self.pre_output_dropout_rate = pre_output_dropout_rate
        self.decoder_num_blocks = decoder_num_blocks
        self.built = False

    def build(self, input_list):
        self.num_feature_maps = len(input_list)
        self.pixel_wise_mlps = nn.ModuleList()
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bicubic", align_corners=True
        )
        self.spatial_mixer = None
        self.closest_square_root = None
        self.num_blocks = len(input_list)
        target_image_size = self.target_image_size

        if len(input_list[0].shape) == 4:
            input_list = [
                self.upsample(x) if x.shape[-1] != target_image_size else x
                for x in input_list
            ]
            input_list = torch.cat(input_list, dim=1)

        elif len(input_list[0].shape) == 3:
            self.rescale_conv = nn.Conv1d(
                input_list[0].shape[1],
                target_image_size * target_image_size,
                kernel_size=1,
                stride=1,
            )

            input_list = [
                self.rescale_conv(x).permute([0, 2, 1])
                if x.shape[1] != target_image_size * target_image_size
                else x.permute(
                    [0, 2, 1]
                )  # (b, sequence, features) -> (b, features, sequence)
                for x in input_list
            ]

            input_list = torch.cat(input_list, dim=1)

        if len(input_list.shape) == 4:
            in_channels = input_list.shape[1]
            self.mlp = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.num_blocks * self.hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm2d(
                    num_features=self.num_blocks * self.hidden_size,
                ),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=self.dropout_rate, inplace=False),
                nn.Conv2d(
                    self.num_blocks * self.hidden_size,
                    self.num_blocks * self.hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm2d(
                    num_features=self.num_blocks * self.hidden_size,
                ),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=self.dropout_rate, inplace=False),
            )

            self.fuse_features = nn.Conv2d(
                self.num_blocks * self.hidden_size,
                self.hidden_size,
                kernel_size=1,
            )
            self.fuse_features_norm = nn.LazyInstanceNorm2d()

            self.fuse_features_act = nn.LeakyReLU()
            self.fuse_features_dropout = nn.Dropout2d(
                p=self.pre_output_dropout_rate, inplace=False
            )
            self.final_conv = nn.Conv2d(
                self.hidden_size, self.num_classes, kernel_size=1
            )
        elif len(input_list.shape) == 3:
            in_channels = input_list.shape[1]
            self.mlp = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    self.num_blocks * self.hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm1d(
                    num_features=self.num_blocks * self.hidden_size
                ),
                nn.LeakyReLU(inplace=False),
                nn.Dropout1d(p=self.dropout_rate, inplace=False),
                nn.Conv1d(
                    self.num_blocks * self.hidden_size,
                    self.num_blocks * self.hidden_size,
                    kernel_size=1,
                ),
                nn.InstanceNorm1d(
                    num_features=self.num_blocks * self.hidden_size
                ),
                nn.LeakyReLU(inplace=False),
                nn.Dropout1d(p=self.dropout_rate, inplace=False),
            )

            self.fuse_features = nn.Conv1d(
                self.num_blocks * self.hidden_size,
                self.hidden_size,
                kernel_size=1,
            )
            self.fuse_features_norm = nn.LazyInstanceNorm1d()

            self.fuse_features_act = nn.LeakyReLU()
            self.fuse_features_dropout = nn.Dropout1d(
                p=self.pre_output_dropout_rate, inplace=False
            )
            self.final_conv = nn.Conv1d(
                self.hidden_size, self.num_classes, kernel_size=1
            )

            sequence_length = input_list.shape[2]
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

        self.built = True

    def forward(self, input_list: list):
        """
        Forward pass of SimpleSegmentationDecoder.

        📮 Passes the input list through pixel-wise MLPs, fuses the features, and upscales the result to the target size.

        :param input_list: List of input tensors, either shape (b, c, h, w) or (b, sequence, features).
        :return: Output tensor representing class feature maps of shape (b, num_classes, target_h, target_w).
        """
        if not self.built:
            self.build(input_list)

        if len(input_list[0].shape) == 4:
            input_list = [
                self.upsample(x)
                if x.shape[-1] != self.target_image_size
                else x
                for x in input_list
            ]
            input_list = torch.cat(input_list, dim=1)

        elif len(input_list[0].shape) == 3:
            input_list = [
                self.rescale_conv(x).permute([0, 2, 1])
                if x.shape[1]
                != self.target_image_size * self.target_image_size
                else x.permute(
                    [0, 2, 1]
                )  # (b, sequence, features) -> (b, features, sequence)
                for x in input_list
            ]
            input_list = torch.cat(input_list, dim=1)

        # print(f"input_feature_maps shape: {input_feature_maps.shape}")
        processed_features = self.mlp(input_list)
        # Concatenate the processed features along the channel dimension
        fused_features = processed_features

        # Fuse the features, apply the final convolution layers, and upscale to target size
        fused_features = self.fuse_features(fused_features)
        fused_norm_features = self.fuse_features_norm(fused_features)
        fused_act_features = self.fuse_features_act(fused_norm_features)
        fused_act_features = self.fuse_features_dropout(fused_act_features)
        class_features = self.final_conv(fused_act_features)

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
    (b, c, new_size)
    output_tensor = F.upsample(
        input_tensor,
        size=(sq_root * sq_root),
    )
    return output_tensor.view(b, c, sq_root, sq_root)


class TransformerSegmentationDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        target_image_size: Union[int, tuple] = (64, 64),
        hidden_size: int = 256,
        pre_output_dropout_rate: float = 0.3,
        dropout_rate: float = 0.5,
        decoder_num_heads: int = 8,
        decoder_num_blocks: int = 4,
    ):
        """
        Initialize the TransformerSegmentationDecoder class for segmentation tasks.

        :param num_classes: An integer representing the number of classes to predict for segmentation.
        :param target_image_size: An optional integer or tuple representing the height and width for the target output size. Default is (64, 64).
        :param hidden_size: An optional integer representing the hidden size for pixel-wise MLP layers. Default is 256.
        :param pre_output_dropout_rate: An optional float representing the dropout rate before the final output. Default is 0.3.
        :param dropout_rate: An optional float representing the dropout rate for the transformer layers. Default is 0.5.
        :param decoder_num_heads: An optional integer representing the number of attention heads in the transformer layers. Default is 8.
        :param decoder_num_blocks: An optional integer representing the number of transformer blocks in the decoder. Default is 4.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.target_image_size = (
            target_image_size
            if isinstance(target_image_size, Tuple)
            or isinstance(target_image_size, List)
            else (target_image_size, target_image_size)
        )
        self.dropout_rate = dropout_rate
        self.pre_output_dropout_rate = pre_output_dropout_rate
        self.decoder_num_heads = decoder_num_heads
        self.decoder_num_blocks = decoder_num_blocks
        self.built = False

    def build(self, input_list):
        hidden_size = self.hidden_size
        num_classes = self.num_classes
        pre_output_dropout_rate = self.pre_output_dropout_rate
        dropout_rate = self.dropout_rate
        num_transformer_blocks = self.decoder_num_blocks
        num_heads = self.decoder_num_heads
        target_image_size = self.target_image_size

        self.pixel_wise_mlps = nn.ModuleList()
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bilinear", align_corners=True
        )
        self.num_feature_maps = len(input_list)
        self.spatial_mixer = None
        self.closest_square_root = None
        self.num_blocks = len(input_list)
        logger.info(f"target image size: {target_image_size}")
        if len(input_list[0].shape) == 4:
            input_list = [
                self.upsample(x) if x.shape[-1] != target_image_size[0] else x
                for x in input_list
            ]
            input_list = torch.cat(input_list, dim=1)

        elif len(input_list[0].shape) == 3:
            self.rescale_conv = nn.Conv1d(
                input_list[0].shape[1],
                target_image_size[0] * target_image_size[1],
                kernel_size=1,
                stride=1,
            )

            input_list = [
                self.rescale_conv(x).permute([0, 2, 1])
                if x.shape[1] != target_image_size[0] * target_image_size[1]
                else x.permute(
                    [0, 2, 1]
                )  # (b, sequence, features) -> (b, features, sequence)
                for x in input_list
            ]

            input_list = torch.cat(input_list, dim=1)

        if len(input_list.shape) == 4:
            in_channels = input_list.shape[1]
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
                nhead=num_heads,
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

        elif len(input_list.shape) == 3:
            in_channels = input_list.shape[1]

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
                nhead=num_heads,
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
        self.built = True

    def forward(self, input_list: list):
        """
        Forward pass of SimpleSegmentationDecoder.

        📮 Passes the input list through pixel-wise MLPs, fuses the features, and upscales the result to the target size.

        :param input_list: List of input tensors, either shape (b, c, h, w) or (b, sequence, features).
        :return: Output tensor representing class feature maps of shape (b, num_classes, target_h, target_w).
        """
        if not self.built:
            self.build(input_list)

        if len(input_list[0].shape) == 4:
            input_list = [
                self.upsample(x)
                if x.shape[-1] != self.target_image_size[0]
                else x
                for x in input_list
            ]
            input_list = torch.cat(input_list, dim=1)

        elif len(input_list[0].shape) == 3:
            input_list = [
                self.rescale_conv(x).permute([0, 2, 1])
                if x.shape[1]
                != self.target_image_size[0] * self.target_image_size[1]
                else x.permute(
                    [0, 2, 1]
                )  # (b, sequence, features) -> (b, features, sequence)
                for x in input_list
            ]
            input_list = torch.cat(input_list, dim=1)

        projected_features = self.projection_layer(input_list)

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
