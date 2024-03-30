import pytest
import torch

from gate.models.backbones import GATEImageTextEncoder
from gate.models.backbones.bert_text import (  # replace 'your_module' with the module where you have defined CLIPAdapter
    BERTCLIPEncoder,
    BertModelPaths,
    CLIPModelPaths,
)


@pytest.fixture
def encoder():
    # You can use a real model name here or mock the CLIPModel and CLIPProcessor as needed
    return BERTCLIPEncoder(
        clip_model_name=CLIPModelPaths.openai_b_16,
        bert_model_name=BertModelPaths.base_uncased,
        image_size=224,
        num_projection_features=64,
    )


def test_clip_adapter_init(encoder: GATEImageTextEncoder):
    assert encoder.text_embedding is not None
    assert encoder.get_transforms is not None


def test_forward_pass_image(encoder):
    # You will need to mock or create a sample image tensor here
    image_tensor = torch.rand((1, 3, 224, 224))  # Mocking an image tensor
    result = encoder.forward(image=image_tensor)
    assert "image" in result
    assert "features" in result["image"]
    assert "raw_features" in result["image"]
    assert "per_layer_raw_features" in result["image"]


def test_forward_pass_text(encoder):
    # You will need to mock or create a sample text tensor here
    text_tensor = torch.randint(0, 2000, (1, 10))  # Mocking a text tensor
    result = encoder.forward(text=text_tensor)
    assert "text" in result
    assert "features" in result["text"]
    assert "raw_features" in result["text"]
    # assert "per_layer_raw_features" in result["text"]


def test_forward_pass_video(encoder):
    # You will need to mock or create a sample video tensor here
    video_tensor = torch.rand((1, 10, 3, 224, 224))  # Mocking a video tensor
    result = encoder.forward(video=video_tensor)
    assert "video" in result
    assert "features" in result["video"]
    assert "raw_features" in result["video"]
    assert "per_layer_raw_features" in result["video"]


def test_forward_pass_raises_exception_with_no_input(encoder):
    with pytest.raises(ValueError):
        encoder.forward()
