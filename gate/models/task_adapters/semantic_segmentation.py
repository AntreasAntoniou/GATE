from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from gate.models.task_adapters import BaseModule
from gate.models.task_adapters.utils import get_similarities


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
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.activation1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.activation2 = nn.GELU()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        out = self.up(out + residual)
        return out


from gate.metrics.segmentation import diff_dice_loss, diff_sigmoid_focal_loss
from gate.metrics.segmentation import (
    miou_loss,
    dice_loss,
    normalized_surface_dice_loss,
    generalized_dice_loss,
    roc_auc_score,
)


def optimization_loss(logits, labels):
    """
    📝 Optimization Loss
    Args:
        logits: (B, C, H, W)
        labels: (B, 1, H, W)
    """

    logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = labels.reshape(-1)
    cross_entropy_loss = F.cross_entropy(logits, labels)
    dice_loss = dice_loss_fn(logits, labels)
    focal_loss = focal_loss_fn(logits, labels)

    return loss


class SegmentationViT(nn.Module):
    """
    Vision Transformer for Semantic Segmentation.
    """

    def __init__(
        self,
        encoder_model: nn.Module,
        embed_dim=768,
        decoder_embed_dim=768,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        num_classes=100,
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
        self.patch_embedding = self.encoder.vision_model.embeddings

        self.decoder_embedding_dimension = decoder_embed_dim
        self.decoder = nn.Linear(
            embed_dim, self.decoder_embedding_dimension, bias=True
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.decoder_embedding_dimension,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.pre_upsample_projection = nn.Conv1d(198, 196, kernel_size=1)
        self.upsample_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    in_channels=self.decoder_embedding_dimension,
                    out_channels=self.decoder_embedding_dimension,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_normalization = norm_layer(
            self.decoder_embedding_dimension
        )
        self.class_decoder = nn.Conv2d(
            self.decoder_embedding_dimension, num_classes, kernel_size=1
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.patch_embedding.num_patches + 1,
                self.decoder_embedding_dimension,
            ),
            requires_grad=False,
        )

        self.init_weights()

    def init_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_position_embedding.shape[-1],
            int(self.patch_embedding.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_position_embedding.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embedding.position_embedding.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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

    def forward(self, image):
        """
            Forward pass for the segmentation model.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Segmentation map.
        """

        batch, _, height, width = image.shape

        features = self.encoder(image)["image"]["raw_features"]
        decoder_inputs = self.decoder(features)

        class_tokens = self.class_token.expand(decoder_inputs.shape[0], -1, -1)
        decoder_inputs += self.decoder_position_embedding
        decoder_inputs = torch.cat((class_tokens, decoder_inputs), dim=1)

        for block in self.decoder_blocks:
            decoder_inputs = block(decoder_inputs)

        decoder_inputs = self.decoder_normalization(decoder_inputs)
        decoder_inputs = self.pre_upsample_projection(decoder_inputs)
        decoder_inputs = decoder_inputs.permute([0, 2, 1])
        batch, channels, sequence = decoder_inputs.shape
        feature_map_size = int(sequence**0.5)
        decoder_inputs = decoder_inputs.view(
            batch, channels, feature_map_size, feature_map_size
        )

        for block in self.upsample_blocks:
            decoder_inputs = block(decoder_inputs)

        decoder_inputs = F.interpolate(
            decoder_inputs, size=(height, width), mode="bilinear"
        )
        output = self.class_decoder(decoder_inputs)

        return output
