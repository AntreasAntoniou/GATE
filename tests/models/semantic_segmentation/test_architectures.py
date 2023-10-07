import pytest
import torch

from gate.models.task_adapters.semantic_segmentation import (
    ChannelMixerDecoder,
    TransformerSegmentationDecoderHead,
)


@pytest.mark.parametrize(
    "model_type", [ChannelMixerDecoder, TransformerSegmentationDecoderHead]
)
def test_forward_pass(model_type):
    # Initialize the model
    model = model_type(num_classes=3, target_image_size=(32, 32))

    # Create synthetic input tensor lists with varying shapes
    input_list_1 = [torch.randn(1, 5, 16, 16), torch.randn(1, 4, 32, 32)]

    # Forward pass
    output_1 = model(input_list_1)

    # Validate output shapes
    assert output_1.shape == (1, 3, 32, 32)
