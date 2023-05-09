import torch
import pytest
from gate.models.backbones.timm import (
    TimmCLIPAdapter,
)  # replace 'your_module' with the module where you have defined CLIPAdapter


def test_TimmCLIPAdapter_resnet():
    clip_model_name = "openai/clip-vit-base-patch32"
    timm_model_name = "timm/resnet50.a1_in1k"
    adapter = TimmCLIPAdapter(
        timm_model_name=timm_model_name, clip_model_name=clip_model_name
    )

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


def test_TimmCLIPAdapter_vit():
    clip_model_name = "openai/clip-vit-base-patch32"
    timm_model_name = "timm/vit_tiny_patch16_224.augreg_in21k"
    adapter = TimmCLIPAdapter(
        timm_model_name=timm_model_name, clip_model_name=clip_model_name
    )

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
