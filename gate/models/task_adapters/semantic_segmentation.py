import math
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
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


from gate.metrics.segmentation import (
    dice_loss,
    diff_dice_loss,
    diff_sigmoid_focal_loss,
    generalized_dice_loss,
    miou_loss,
    roc_auc_score,
)


def metrics(logits, labels, label_dim, num_classes):
    logits: torch.Tensor = logits.detach().float()
    labels: torch.Tensor = labels.detach()
    logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = labels.reshape(-1)
    return {
        "roc_auc_score": torch.tensor(
            roc_auc_score(
                logits, labels, num_classes=logits.shape[1], label_dim=1
            )
        ),
        "miou_loss": torch.tensor(
            miou_loss(logits, labels, num_classes=logits.shape[1], label_dim=1)
        ),
        "dice_loss": torch.tensor(
            dice_loss(logits, labels, num_classes=logits.shape[1], label_dim=1)
        ),
        "generalized_dice_loss": torch.tensor(
            generalized_dice_loss(
                logits, labels, num_classes=logits.shape[1], label_dim=1
            )
        ),
    }


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

    loss = cross_entropy_loss + dice_loss + focal_loss

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        max_len = x.shape[1]
        d_model = x.shape[2]
        position = torch.arange(max_len).unsqueeze(1).to(x.device)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        ).to(x.device)
        pe = torch.zeros(1, max_len, d_model).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term).to(x.device)
        pe[0, :, 1::2] = torch.cos(position * div_term).to(x.device)
        x = x + pe[: x.size(0)]
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
        decoder_depth: int = 1,
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

        self.pre_upsample_projection = nn.Conv1d(
            self.num_patches + 1,
            int(math.floor(math.sqrt(self.num_patches))) ** 2,
            kernel_size=1,
        )

        self.upsample_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    in_channels=self.decoder_embedding_dimension,
                    out_channels=self.decoder_embedding_dimension,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.additional_projection = None
        self.decoder_normalization = norm_layer(
            self.decoder_embedding_dimension
        )
        self.class_decoder = nn.Conv2d(
            self.decoder_embedding_dimension, num_classes, kernel_size=1
        )

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
        output = {}
        if labels is not None:
            output = optimization_loss(logits, labels)
            # output |= metrics(
            #     logits,
            #     labels,
            #     label_dim=1,
            #     num_classes=self.num_classes,
            # )
            # print(output)

        return output

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
        decoder_inputs = self.decoder(features)

        class_tokens = self.class_token.expand(decoder_inputs.shape[0], -1, -1)
        decoder_inputs = self.positional_encoding(decoder_inputs)
        decoder_inputs = torch.cat((class_tokens, decoder_inputs), dim=1)

        for block in self.decoder_blocks:
            decoder_inputs = block(decoder_inputs)

        decoder_inputs = self.decoder_normalization(decoder_inputs)
        print(f"stem decoder_inputs.shape: {decoder_inputs.shape}")
        if decoder_inputs.shape[1] != self.num_patches + 1:
            if self.additional_projection is None:
                self.additional_projection = nn.Conv1d(
                    decoder_inputs.shape[1],
                    self.num_patches + 1,
                    kernel_size=1,
                ).to(decoder_inputs.device)
            decoder_inputs = self.additional_projection(decoder_inputs)
            print(
                f"additional projection decoder_inputs.shape: {decoder_inputs.shape}"
            )

        decoder_inputs = self.pre_upsample_projection(decoder_inputs)
        print(
            f"pre upsample projection decoder_inputs.shape: {decoder_inputs.shape}"
        )
        decoder_inputs = decoder_inputs.permute([0, 2, 1])
        batch, channels, sequence = decoder_inputs.shape
        feature_map_size = int(sequence**0.5)
        decoder_inputs = decoder_inputs.view(
            batch, channels, feature_map_size, feature_map_size
        )
        print(f"reshape decoder_inputs.shape: {decoder_inputs.shape}")

        for block in self.upsample_blocks:
            decoder_inputs = block(decoder_inputs)
            print(
                f"upsample block decoder_inputs.shape: {decoder_inputs.shape}"
            )

        decoder_inputs = F.interpolate(
            decoder_inputs, size=(height, width), mode="bilinear"
        )
        print(f"interpolate decoder_inputs.shape: {decoder_inputs.shape}")
        output = self.class_decoder(decoder_inputs)
        print(f"class decoder output.shape: {output.shape}")
        output = {"logits": output}
        print(f"output: {output.shape}")

        if return_loss_and_metrics:
            output |= self.compute_loss_and_metrics(
                logits=output["logits"], labels=labels
            )

        return output
