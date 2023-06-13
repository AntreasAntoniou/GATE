from dataclasses import dataclass

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
        # print(x.shape)
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


import torch
from torch import nn, Tensor
import math


def interpolate_position_encoding(
    pos_embed: Tensor, x: Tensor, w: int, h: int, patch_size: int
) -> Tensor:
    """Interpolate the position encoding based on the input tensor dimensions.

    This code is adapted from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174

    Args:
        pos_embed (Tensor): Position embedding tensor (B, N+1, D).
        x (Tensor)       : Input tensor (B, N+1, D).
        w (int)          : Width of the input.
        h (int)          : Height of the input.
        patch_size (int) : Patch size used in the patch embedding module.

    Returns:
        Tensor: Interpolated position encoding tensor (B, npatch+1, D).
    """

    N = pos_embed.shape[1] - 1
    npatch = (w // patch_size) * (h // patch_size)

    if npatch == N:
        return pos_embed

    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]

    w0 = w // patch_size
    h0 = h // patch_size
    w0, h0 = (
        w0 + 0.1,
        h0 + 0.1,
    )  # Add a small number to avoid floating point errors

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )

    assert (
        int(w0) == patch_pos_embed.shape[-2]
        and int(h0) == patch_pos_embed.shape[-1]
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
