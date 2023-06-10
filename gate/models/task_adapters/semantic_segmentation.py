import math
from functools import partial
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from timm.models.vision_transformer import Block
from transformers import SamMaskDecoderConfig, SamModel
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


class ResidualConvBlock(nn.Module):
    """
    📝 Residual Convolutional Block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=5, stride=2
        )
        self.activation1 = nn.GELU()
        self.norm1 = nn.InstanceNorm2d(out_channels)

        self.conv2 = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=1,
        )
        self.activation2 = nn.GELU()
        self.norm2 = nn.InstanceNorm2d(out_channels)

        self.up1 = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.activation1(out)
        out = self.norm1(out)

        if self.up1 is None:
            self.up1 = nn.Upsample(
                size=(out.shape[2], out.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        out = self.up1(residual) + out

        out = self.conv2(out)
        out = self.activation2(out)
        out = self.norm2(out)

        return out


from gate.metrics.segmentation import (
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
    cross_entropy_loss = F.cross_entropy(logits, labels)
    dice_loss = diff_dice_loss(logits, labels)
    focal_loss = diff_sigmoid_focal_loss(logits, labels)

    loss = dice_loss + focal_loss + cross_entropy_loss

    return {
        "loss": loss,
        "cross_entropy_loss": cross_entropy_loss,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
    }


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

            self.positional_encoding = pe

        self.positional_encoding = self.positional_encoding.to(x.device)
        x = x + self.positional_encoding[: x.shape[0]]
        return x


class SegmentationViT(nn.Module):
    """
    Vision Transformer for Semantic Segmentation.
    """

    def __init__(
        self,
        encoder_model: nn.Module,
        model_type: str = "vit",
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: int = 4.0,
        norm_layer: int = nn.LayerNorm,
        num_classes: int = 100,
        num_patches: int = 14,
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
        self.positional_encoding = PositionalEncoding()

        self.decoder_embedding_dimension = decoder_embed_dim
        self.decoder_feature_matcher = nn.Linear(
            embed_dim, self.decoder_embedding_dimension, bias=True
        )
        self.decoder_spatial_matcher = None

        self.channel_projection = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=256,
            kernel_size=1,
        )
        self.decoder_config = SamMaskDecoderConfig(
            num_multimask_outputs=num_classes
        )
        self.decoder = SamMaskDecoder(config=self.decoder_config)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.class_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def compute_loss_and_metrics(
        self, logits, labels: Optional[torch.Tensor] = None
    ):
        output_dict = {}
        if labels is not None:
            output_dict = optimization_loss(logits, labels)
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

        batch, _, height, width = image.shape
        features = self.encoder(image)["image"]["raw_features"]

        decoder_inputs = self.decoder_feature_matcher(features)
        class_tokens = self.class_token.expand(decoder_inputs.shape[0], -1, -1)
        decoder_inputs = torch.cat((class_tokens, decoder_inputs), dim=1)

        if self.decoder_spatial_matcher is None:
            self.decoder_spatial_matcher = nn.Conv1d(
                in_channels=decoder_inputs.shape[1],
                out_channels=int(
                    math.floor(math.sqrt(decoder_inputs.shape[1]))
                )
                ** 2,
                kernel_size=1,
            )
        decoder_inputs = self.decoder_spatial_matcher(decoder_inputs)

        decoder_inputs = decoder_inputs.permute([0, 2, 1])
        batch, channels, sequence = decoder_inputs.shape
        feature_map_size = int(sequence**0.5)
        decoder_inputs = decoder_inputs.view(
            batch, channels, feature_map_size, feature_map_size
        )
        decoder_inputs = self.channel_projection(decoder_inputs)
        decoder_inputs = F.interpolate(decoder_inputs, size=(64, 64))
        decoder_inputs = self.positional_encoding(decoder_inputs)

        # torch.Size([1, 1, 2, 256]),
        # dense_embeddings: torch.Size([1, 256, 64, 64]),
        # image_embeddings: torch.Size([1, 256, 64, 64]),
        # image_positional_embeddings: torch.Size([1, 256, 64, 64])

        print(
            f"position: {self.positional_encoding.positional_encoding.shape}, "
            f"image_embeddings: {decoder_inputs.shape}, dense_embeddings: {decoder_inputs.shape}, "
            f"image_positional_embeddings: {self.positional_encoding.positional_encoding.shape}"
        )

        mask_predictions, _, _ = self.decoder(
            image_embeddings=decoder_inputs,
            image_positional_embeddings=self.positional_encoding.positional_encoding,
            sparse_prompt_embeddings=torch.zeros(
                decoder_inputs.shape[0], 1, 1, 256
            ).to(decoder_inputs.device),
            dense_prompt_embeddings=torch.zeros(decoder_inputs.shape).to(
                decoder_inputs.device
            ),
            multimask_output=False,
            output_attentions=False,
        )

        print(f"mask_predictions: {mask_predictions.shape}")

        output = {
            "logits": mask_predictions,
        }

        print(mask_predictions)

        if return_loss_and_metrics:
            output |= self.compute_loss_and_metrics(
                logits=output["logits"], labels=labels
            )

        return output
