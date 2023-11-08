import pytest
import torch
from rich import print

from gate.models.blocks.segmentation import (
    ChannelMixerDecoder,
    TransformerSegmentationDecoder,
    upsample_tensor,
)


@pytest.mark.parametrize(
    "model_type", [ChannelMixerDecoder, TransformerSegmentationDecoder]
)
def test_forward_pass(model_type):
    # Initialize the model
    print(model_type.__text_signature__)
    model = model_type(num_classes=100)

    # Create synthetic input tensor lists with varying shapes
    input_list = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

    # Forward pass
    output = model(input_list)

    # Validate output shape
    assert output.shape == (
        1,
        100,
        64,
        64,
    ), f"Expected output shape to be (1, 100, 256, 256), got {output.shape}"

    # Validate that the model is built
    assert (
        model.built == True
    ), "Expected model.built to be True, but got False"


def test_upsample_tensor():
    # Create a dummy tensor of shape (batch_size, channels, seq_length)
    input_tensor = torch.randn(
        4, 16, 64
    )  # Batch size = 4, Channels = 16, Seq_length = 64
    output_tensor = upsample_tensor(input_tensor)

    # Validate output shape
    assert output_tensor.shape == (
        4,
        16,
        8,
        8,
    ), f"Expected output shape to be (4, 16, 8, 8), got {output_tensor.shape}"
