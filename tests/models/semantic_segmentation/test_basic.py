import pytest
import torch

from gate.models.backbones.clip import CLIPAdapter
from gate.models.task_adapters.semantic_segmentation import (
    SimpleSegmentationDecoder,
)


def test_basic():
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(low=0, high=100, size=(2, 1, 224, 224))

    encoder_model = CLIPAdapter(model_name="openai/clip-vit-base-patch16")
    model = SimpleSegmentationDecoder(
        encoder_model=encoder_model, num_classes=100
    )

    out = model(x)
    print(out["logits"].shape)
