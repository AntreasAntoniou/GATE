import torch
import pytest
from gate.data.transforms.segmentation_transforms import SameRandomCrop


def test_same_random_crop():
    transform = SameRandomCrop(50)

    # Create a pair of 100x100 images
    img1 = torch.rand(3, 100, 100)
    img2 = torch.rand(3, 100, 100)

    # Apply the transform
    img1_cropped, img2_cropped = transform(img1, img2)

    # Check that both images have been cropped to the specified size
    assert img1_cropped.shape == (3, 50, 50)
    assert img2_cropped.shape == (3, 50, 50)

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
