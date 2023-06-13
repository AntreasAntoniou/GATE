import torch
import pytest

from gate.models.backbones import interpolate_position_encoding


class DummyPatchEmbed(torch.nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size


@pytest.mark.parametrize("w, h", [(14, 14), (28, 28), (112, 112)])
def test_interpolate_position_encoding(w, h):
    B, N, D = 1, 64, 384
    x = torch.rand(B, N + 1, D)
    pos_embed = torch.rand(B, N + 1, D)
    patch_size = 16

    output_pos_embed = interpolate_position_encoding(
        pos_embed, x, w, h, patch_size
    )

    npatch = (w // patch_size) * (h // patch_size)

    assert output_pos_embed.shape == (B, npatch + 1, D)
