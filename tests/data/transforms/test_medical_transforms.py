import pytest
import torch

from gate.data.transforms.segmentation_transforms import (
    MedicalImageSegmentationTransforms,
)


def test_MedicalImageSegmentationTransforms():
    # Initialize the transform class
    transformer = MedicalImageSegmentationTransforms()

    # Create dummy image and mask tensors
    img = torch.rand((2, 3, 224, 224))
    mask = torch.randint(0, 2, (2, 3, 224, 224), dtype=torch.float)

    # Apply the transformations
    transformed_img, transformed_mask = transformer(img, mask)

    # Validate the output shape
    assert (
        transformed_img.shape == img.shape
    ), f"Expected shape {img.shape}, but got {transformed_img.shape}"
    assert (
        transformed_mask.shape == mask.shape
    ), f"Expected shape {mask.shape}, but got {transformed_mask.shape}"

    # Validate the output type
    assert isinstance(
        transformed_img, torch.Tensor
    ), f"Expected output to be a torch.Tensor, but got {type(transformed_img)}"
    assert isinstance(
        transformed_mask, torch.Tensor
    ), f"Expected output to be a torch.Tensor, but got {type(transformed_mask)}"

    # Validate the output range for image
    assert (
        transformed_img.min() >= 0
    ), f"Minimum value in transformed image is less than 0: {transformed_img.min()}"
    assert (
        transformed_img.max() <= 1
    ), f"Maximum value in transformed image is greater than 1: {transformed_img.max()}"

    # Validate the output range for mask
    assert (
        transformed_mask.min() >= 0
    ), f"Minimum value in transformed mask is less than 0: {transformed_mask.min()}"
    assert (
        transformed_mask.max() <= 1
    ), f"Maximum value in transformed mask is greater than 1: {transformed_mask.max()}"
