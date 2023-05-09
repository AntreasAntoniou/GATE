import torch
import pytest
from gate.models.backbones.clip import (
    CLIPAdapter,
)  # replace 'your_module' with the module where you have defined CLIPAdapter


def test_CLIPAdapter():
    model_name = "openai/clip-vit-base-patch32"
    adapter = CLIPAdapter(model_name)

    # Test with only image input
    image = torch.randn(1, 3, 224, 224)
    output = adapter.forward(image=image)
    assert "image_features" in output
    assert "image_projection_output" in output
    assert "text_features" not in output
    assert "text_projection_output" not in output

    # Test with only text input
    text = torch.randint(0, 100, (1, 10))  # Assuming text input is token IDs
    output = adapter.forward(text=text)
    assert "text_features" in output
    assert "text_projection_output" in output
    assert "image_features" not in output
    assert "image_projection_output" not in output

    # Test with both image and text input
    output = adapter.forward(image=image, text=text)
    assert "image_features" in output
    assert "image_projection_output" in output
    assert "text_features" in output
    assert "text_projection_output" in output

    # Test with neither image nor text input
    with pytest.raises(ValueError):
        output = adapter.forward()
