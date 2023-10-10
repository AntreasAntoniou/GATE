import pytest
import torch

from gate.models.task_adapters.semantic_segmentation import (
    ChannelMixerDecoder,
    TransformerSegmentationDecoder,
    upsample_tensor,
)


@pytest.mark.parametrize(
    "model_type", [ChannelMixerDecoder, TransformerSegmentationDecoder]
)
def test_forward_pass(model_type):
    # Initialize the model
    model = model_type(num_classes=3, target_image_size=(32, 32))

    # Create synthetic input tensor lists with varying shapes
    input_list = [torch.randn(1, 5, 16, 16), torch.randn(1, 4, 32, 32)]

    # Forward pass
    output = model(input_list)

    # Validate output shape
    assert output.shape == (
        1,
        3,
        32,
        32,
    ), f"Expected output shape to be (4, 10, 128, 128), got {output.shape}"

    # Validate that the model is built
    assert (
        model.is_built == True
    ), "Expected model.is_built to be True, but got False"


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
