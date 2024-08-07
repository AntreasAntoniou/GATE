import numpy as np
import pytest
import torch
from PIL import Image

from gate.data.transforms.segmentation import DualImageRandomCrop


def test_same_random_crop():
    transform = DualImageRandomCrop(50)

    # Create a pair of 100x100 images as PyTorch Tensors
    img1 = torch.rand(3, 100, 100)
    img2 = torch.rand(3, 100, 100)

    # Apply the transform
    img1_cropped, img2_cropped = transform(img1, img2)

    # Check that both images have been cropped to the specified size
    assert img1_cropped.shape == (3, 50, 50)
    assert img2_cropped.shape == (3, 50, 50)

    # Repeat the test with PIL Images
    # Create images with pixel values ranging from 0 to 255 (8-bit unsigned integers)
    img1 = Image.fromarray(
        np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    )
    img2 = Image.fromarray(
        np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    )

    img1_cropped, img2_cropped = transform(img1, img2)

    assert img1_cropped.size == (50, 50)
    assert img2_cropped.size == (50, 50)

    # Check an edge case where the images are smaller than the crop size
    img1 = torch.rand(3, 30, 30)
    img2 = torch.rand(3, 30, 30)

    with pytest.raises(ValueError):
        # This should raise an error because the images are too small to be cropped to 50x50
        img1_cropped, img2_cropped = transform(img1, img2)

    # Check that an error is raised when the input images have different dimensions
    img1 = torch.rand(3, 100, 100)
    img2 = torch.rand(3, 100, 90)

    with pytest.raises(ValueError):
        # This should raise an error because the images have different dimensions
        img1_cropped, img2_cropped = transform(img1, img2)

    # Check that an error is raised when the input images are not PyTorch Tensors or PIL Images
    img1 = np.random.rand(3, 100, 100)
    img2 = np.random.rand(3, 100, 100)

    with pytest.raises(TypeError):
        # This should raise an error because the images are neither PyTorch Tensors nor PIL Images
        img1_cropped, img2_cropped = transform(img1, img2)
