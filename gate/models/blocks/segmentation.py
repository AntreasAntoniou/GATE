import math
import time
from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


class ChannelMixerDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        target_image_size: Union[int, tuple] = (64, 64),
        hidden_size: int = 256,
        decoder_num_blocks: int = 2,
        pre_output_dropout_rate: float = 0.3,
        dropout_rate: float = 0.5,
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
        self.decoder_num_blocks = decoder_num_blocks
        self.pre_output_dropout_rate = pre_output_dropout_rate
        self.is_built = False
        self.weighted_upscale = nn.LazyConv1d(
            self.target_image_size * self.target_image_size, 1
        )

    def build(self, input_list: List[torch.Tensor]):
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bilinear", align_corners=True
        )
        input_list = (
            [self.upsample(x) for x in input_list]
            if len(input_list[0].shape) == 4
            else [
                rearrange(self.weighted_upscale(x), "b l c -> b c l")
                for x in input_list
            ]
        )
        out = torch.cat(input_list, dim=1)
        input_shape = out.shape

        norm = (
            nn.LazyInstanceNorm2d
            if len(input_shape) == 4
            else nn.LazyInstanceNorm1d
        )
        conv = nn.LazyConv2d if len(input_shape) == 4 else nn.LazyConv1d
        dropout = nn.Dropout2d if len(input_shape) == 4 else nn.Dropout1d
        act = nn.LeakyReLU

        mlp_layers = []

        for _ in range(self.decoder_num_blocks):
            mlp_layers.extend(
                [
                    conv(out_channels=self.hidden_size, kernel_size=1),
                    norm(),
                    act(),
                    dropout(self.dropout_rate),
                ]
            )

        self.mlp = nn.Sequential(
            *mlp_layers,
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
        input_list = (
            [self.upsample(x) for x in input_list]
            if len(input_list[0].shape) == 4
            else [
                rearrange(self.weighted_upscale(x), "b l c -> b c l")
                for x in input_list
            ]
        )

        input_feature_maps = torch.cat(input_list, dim=1)
        processed_features = self.mlp(input_feature_maps)
        fused_features = self.fuse_features(processed_features)
        if len(fused_features.shape) == 3:
            fused_features = rearrange(
                fused_features,
                "b c (h w) -> b c h w",
                h=self.target_image_size,
                w=self.target_image_size,
            )
        class_features = self.final_conv(fused_features)

        return class_features


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
        super().__init__()
        self.num_classes = num_classes
        self.target_image_size = (
            target_image_size
            if isinstance(target_image_size, int)
            else target_image_size[0]
        )
        self.decoder_num_heads = decoder_num_heads
        self.decoder_num_blocks = decoder_num_blocks
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.pre_output_dropout_rate = pre_output_dropout_rate
        self.upsample = nn.Upsample(
            size=self.target_image_size, mode="bilinear", align_corners=True
        )
        self.weighted_upscale = nn.LazyConv1d(
            self.target_image_size * self.target_image_size, 1
        )
        self.is_built = False

    def build(self, input_shape: tuple):
        self.projection_layer = (
            nn.Conv2d(input_shape[1], self.hidden_size, 1)
            if len(input_shape) == 4
            else nn.Conv1d(input_shape[1], self.hidden_size, 1)
        )

        self.fuse_features = (
            nn.Sequential(
                nn.Conv2d(self.hidden_size, self.hidden_size, 1),
                nn.InstanceNorm2d(self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout2d(self.pre_output_dropout_rate),
            )
            if len(input_shape) == 4
            else nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, 1),
                nn.InstanceNorm1d(self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout1d(self.pre_output_dropout_rate),
            )
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.decoder_num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.segmentation_processing_head = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=self.decoder_num_blocks,
            norm=nn.LayerNorm(self.hidden_size),
        )

        self.final_conv = nn.Conv2d(self.hidden_size, self.num_classes, 1)
        self.is_built = True

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        input_list = (
            [self.upsample(x) for x in input_list]
            if len(input_list[0].shape) == 4
            else [
                rearrange(self.weighted_upscale(x), "b l c -> b c l")
                for x in input_list
            ]
        )

        input_feature_maps = torch.cat(input_list, dim=1)

        if not self.is_built:
            self.build(input_feature_maps.shape)

        projected_features = self.projection_layer(input_feature_maps)

        fused_features = self.fuse_features(projected_features)

        fused_features = (
            rearrange(fused_features, "b c h w -> b (h w) c")
            if len(fused_features.shape) == 4
            else rearrange(fused_features, "b c l -> b (l) c")
        )

        transformed_features = self.segmentation_processing_head(
            fused_features
        )

        if transformed_features.shape[1] != (self.target_image_size) ** 2:
            transformed_features = F.adaptive_avg_pool1d(
                rearrange(transformed_features, "b l c -> b c l"),
                (self.target_image_size) ** 2,
            )
            transformed_features = rearrange(
                transformed_features,
                "b c (h w) -> b (h w) c",
                h=self.target_image_size,
                w=self.target_image_size,
            )

        transformed_features = rearrange(
            transformed_features,
            "b (h w) c -> b c h w",
            h=self.target_image_size,
            w=self.target_image_size,
        )
        class_features = self.final_conv(transformed_features)

        return class_features
