import pytest
import torch

from gate.models.backbones import interpolate_position_encoding


@pytest.mark.parametrize(
    "w, h, patch_size, class_token",
    [(16, 16, 4, True), (32, 32, 8, True), (32, 32, 8, False)],
)
def test_interpolate_position_encoding(w, h, patch_size, class_token):
    B, N, D = 2, 16, 384
    input_shape = (B, N + int(class_token), D)
    pos_embed = torch.rand(input_shape)
    x = torch.rand(
        B, N, D
    )  # simplified input tensor without considering spatial dimensions

    class_token_idx = 0 if class_token else None
    output_pos_embed = interpolate_position_encoding(
        pos_embed, x, w, h, patch_size, B, class_token_idx
    )

    npatch = (w // patch_size) * (h // patch_size)
    output_shape = (B, npatch + int(class_token), D)

    assert (
        output_pos_embed.shape == output_shape
    ), f"Expected shape: {output_shape}, but got: {output_pos_embed.shape}"
