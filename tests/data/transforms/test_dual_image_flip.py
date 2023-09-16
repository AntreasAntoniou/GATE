# Pytest functions
import numpy as np
import pytest
import torch
from PIL import Image

from gate.data.transforms.segmentation_transforms import DualImageRandomFlip


def test_flip_tensor():
    img1 = torch.rand((3, 224, 224))
    img2 = torch.rand((3, 224, 224))
    flipper = DualImageRandomFlip(p=1)  # Setting p=1 to ensure flip happens
    flipped_img1, flipped_img2 = flipper(img1, img2)
    assert torch.equal(img1, torch.flip(flipped_img1, [-1]))
    assert torch.equal(img2, torch.flip(flipped_img2, [-1]))


def test_flip_image():
    img1 = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    img2 = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    flipper = DualImageRandomFlip(p=1)  # Setting p=1 to ensure flip happens
    flipped_img1, flipped_img2 = flipper(img1, img2)
    assert (
        np.array(img1).all()
        == np.array(flipped_img1.transpose(Image.FLIP_LEFT_RIGHT)).all()
    )
    assert (
        np.array(img2).all()
        == np.array(flipped_img2.transpose(Image.FLIP_LEFT_RIGHT)).all()
    )


def test_mismatched_dims():
    img1 = torch.rand((3, 224, 224))
    img2 = torch.rand((3, 200, 200))
    flipper = DualImageRandomFlip()
    with pytest.raises(ValueError):
        flipper(img1, img2)


def test_invalid_input_type():
    img1 = torch.rand((3, 224, 224))
    img2 = "InvalidType"
    flipper = DualImageRandomFlip()
    with pytest.raises(TypeError):
        flipper(img1, img2)
