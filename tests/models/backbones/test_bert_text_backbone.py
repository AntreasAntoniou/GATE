import pytest
import torch

from gate.models.backbones.bert_text import (  # replace 'your_module' with the module where you have defined CLIPAdapter
    BertAdapter,
    BertModelPaths,
    CLIPModelPaths,
)


@pytest.fixture
def clip_adapter():
    # You can use a real model name here or mock the CLIPModel and CLIPProcessor as needed
    return BertAdapter(
        clip_model_name=CLIPModelPaths.openai_b_16,
        bert_model_name=BertModelPaths.base_uncased,
        image_size=224,
    )


def test_clip_adapter_init(clip_adapter):
    assert clip_adapter.clip is not None
    assert clip_adapter.text_transforms is not None


def test_forward_pass_image(clip_adapter):
    # You will need to mock or create a sample image tensor here
    image_tensor = torch.rand((1, 3, 224, 224))  # Mocking an image tensor
    result = clip_adapter.forward(image=image_tensor)
    assert "image" in result
    assert "classifier" in result["image"]


# def test_forward_pass_text(clip_adapter):
#     # You will need to mock or create a sample text tensor here
#     text_tensor = torch.randint(0, 2000, (1, 10))  # Mocking a text tensor
#     result = clip_adapter.forward(text=text_tensor)
#     assert "text" in result
#     assert "projection_output" in result["text"]


# def test_forward_pass_video(clip_adapter):
#     # You will need to mock or create a sample video tensor here
#     video_tensor = torch.rand((1, 10, 3, 224, 224))  # Mocking a video tensor
#     result = clip_adapter.forward(video=video_tensor)
#     assert "video" in result
#     assert "projection_output" in result["video"]


# def test_forward_pass_raises_exception_with_no_input(clip_adapter):
#     with pytest.raises(ValueError):
#         clip_adapter.forward()
