from dataclasses import dataclass
from typing import Optional

import PIL
import torch
import torchvision.transforms as T

single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))


def image_dim_reshape(x):
    if len(x.shape) == 5:
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])

    return x


@dataclass
class Modality:
    image: str = "image"
    text: str = "text"
    audio: str = "audio"
    video: str = "video"


def apply_preprocessing_transforms(transforms, x, modality=Modality.image):
    input_shape = None
    is_5d_tensor = False
    if isinstance(x, PIL.Image.Image) and modality == Modality.image:
        x = T.ToTensor()(x)
        if x.shape[0] == 1:
            x = single_to_three_channel(x)
        x = T.ToPILImage()(x)

    if isinstance(x, torch.Tensor) and modality == Modality.image:
        input_shape = x.shape
        is_5d_tensor = len(x.shape) == 5
        x = image_dim_reshape(x)

    if transforms is not None:
        x = transforms(x)

    if (
        input_shape is not None
        and isinstance(x, torch.Tensor)
        and is_5d_tensor
    ):
        x = x.view(input_shape[0], input_shape[1], *x.shape[1:])

    return x


import math

import torch
from torch import Tensor, nn


def interpolate_position_encoding(
    pos_embed: Tensor,
    x: Tensor,
    w: int,
    h: int,
    patch_size: int,
    batch_size: int,
    class_token_idx: Optional[int] = None,
) -> Tensor:
    """
    Interpolate the position encoding based on the input tensor dimensions.

    Args:
        pos_embed (torch.Tensor): Position embedding tensor.
            Shape: (B, N, D) or (B, N+1, D), depending on the class token.
        x (torch.Tensor): Input tensor.
            Shape: Either (B, N, D) or (B, C, H, W).
        w (int): Width of the input.
        h (int): Height of the input.
        patch_size (int): Patch size used in the patch embedding module.
        batch_size (int): The batch size, required to process the data.
        class_token_idx (Optional[int], optional): Index of the class token,
            if present. Defaults to None.

    Returns:
        torch.Tensor: Interpolated position encoding tensor.
            Shape: (B, npatch+1, D) or (B, npatch, D), depending on the class token.
    """

    N = pos_embed.shape[1] - (1 if class_token_idx is not None else 0)
    npatch = (w // patch_size) * (h // patch_size)

    if npatch == N:
        return pos_embed

    # Separate class and patch position embeddings
    if class_token_idx is not None:
        class_pos_embed = pos_embed[:, class_token_idx]
        patch_pos_embed = torch.cat(
            (
                pos_embed[:, :class_token_idx],
                pos_embed[:, class_token_idx + 1 :],
            ),
            dim=1,
        )
    else:
        class_pos_embed = None
        patch_pos_embed = pos_embed

    dim = x.shape[-1]
    w0, h0 = w // patch_size, h // patch_size

    # Add a small number to avoid floating point errors ⚠️
    w0, h0 = w0 + 0.1, h0 + 0.1

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(
        batch_size, -1, dim
    )

    # Add the class position embedding back if it was present
    if class_token_idx is not None:
        return torch.cat(
            (class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1
        )

    return patch_pos_embed
