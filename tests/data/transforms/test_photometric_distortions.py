# Importing pytest and required libraries for testing
import numpy as np
import pytest
import torch
from PIL import Image

from gate.data.transforms.segmentation import PhotoMetricDistortion


# Function for generating test image data
def generate_test_image(img_type):
    if img_type == "tensor":
        return torch.rand((3, 224, 224))
    elif img_type == "numpy":
        return np.random.rand(224, 224, 3)
    elif img_type == "PIL":
        return Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )
    else:
        return None


# Test case for PhotoMetricDistortion
def test_photo_metric_distortion():
    transformer = PhotoMetricDistortion()

    for img_type in ["tensor", "numpy", "PIL"]:
        img = generate_test_image(img_type)
        transformed_img = transformer(img)

        assert isinstance(
            transformed_img, torch.Tensor
        ), f"Output should be a torch.Tensor, got {type(transformed_img)} instead."
        assert transformed_img.shape == torch.Size(
            [3, 224, 224]
        ), f"Unexpected output shape {transformed_img.shape}."
        assert (
            transformed_img.min() >= 0 and transformed_img.max() <= 1
        ), "Pixel values should be between 0 and 1."

        if img_type == "tensor":
            assert not torch.equal(
                img, transformed_img
            ), "The output tensor should not be the same as the input tensor."

        print(f"{img_type} test passed.")
