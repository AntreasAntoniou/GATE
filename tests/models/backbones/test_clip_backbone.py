import pytest
import torch

from gate.models.backbones.clip import (  # replace 'your_module' with the module where you have defined CLIPAdapter
    CLIPAdapter,
)


def test_CLIPAdapter():
    model_name = "openai/clip-vit-base-patch32"
    adapter = CLIPAdapter(model_name)

    # Test with only image input
    image = torch.randn(1, 3, 224, 224)
    output = adapter.forward(image=image)

    # Test with only text input
    text = torch.randint(0, 100, (1, 10))  # Assuming text input is token IDs
    output = adapter.forward(text=text)

    # Test with both image and text input
    output = adapter.forward(image=image, text=text)

    # Test with neither image nor text input
    with pytest.raises(ValueError):
        output = adapter.forward()
