import pytest
import torch

from gate.models.backbones.clip_image import (  # replace 'your_module' with the module where you have defined CLIPAdapter
    CLIPModelPaths,
    CLIPVisionAdapter,
)


@pytest.fixture
def adapter():
    # You can use a real model name here or mock the CLIPModel and CLIPProcessor as needed
    return CLIPVisionAdapter(model_name=CLIPModelPaths.openai_b_16)


def test_clip_adapter_init(adapter):
    assert adapter.clip is not None
    assert adapter.text_transforms is not None


def test_forward_pass_image(adapter):
    # You will need to mock or create a sample image tensor here
    image_tensor = torch.rand((1, 3, 224, 224))  # Mocking an image tensor
    result = adapter.forward(image=image_tensor)
    assert "image" in result
    assert "classifier" in result["image"]
    assert "raw_features" in result["image"]
    assert "per_layer_raw_features" in result["image"]


def test_forward_pass_text(adapter):
    # You will need to mock or create a sample text tensor here
    text_tensor = torch.randint(0, 2000, (1, 10))  # Mocking a text tensor
    result = adapter.forward(text=text_tensor)
    assert "text" in result
    assert "classifier" in result["text"]
    assert "raw_features" in result["text"]
    assert "per_layer_raw_features" in result["text"]


def test_forward_pass_video(adapter):
    # You will need to mock or create a sample video tensor here
    video_tensor = torch.rand((1, 10, 3, 224, 224))  # Mocking a video tensor
    result = adapter.forward(video=video_tensor)
    assert "video" in result
    assert "classifier" in result["video"]
    assert "raw_features" in result["video"]
    assert "per_layer_raw_features" in result["video"]


def test_forward_pass_raises_exception_with_no_input(adapter):
    with pytest.raises(ValueError):
        adapter.forward()
