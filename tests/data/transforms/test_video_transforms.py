import random

import pytest
import torch
from torchvision.transforms import functional as F


def test_TemporalCrop():
    transform = TemporalCrop((200, 200))
    input_dict = {"video": torch.rand(2, 3, 8, 224, 224)}
    output_dict = transform(input_dict)
    assert output_dict["video"].shape[-2:] == (
        200,
        200,
    ), "TemporalCrop did not correctly crop the video."


def test_TemporalFlip():
    transform = TemporalFlip()
    input_dict = {"video": torch.rand(2, 3, 8, 224, 224)}
    output_dict = transform(input_dict)
    if random.random() < 0.5:
        assert torch.equal(
            input_dict["video"], output_dict["video"]
        ), "TemporalFlip modified the video despite probability check."
    else:
        assert torch.equal(
            input_dict["video"].flip(-1), output_dict["video"]
        ), "TemporalFlip did not correctly flip the video."


def test_TemporalRotation():
    transform = TemporalRotation()
    input_dict = {"video": torch.rand(2, 3, 8, 224, 224)}
    output_dict = transform(input_dict)
    # Add assertions to check if the video has been rotated correctly


def test_TemporalBrightnessContrast():
    transform = TemporalBrightnessContrast()
    input_dict = {"video": torch.rand(2, 3, 8, 224, 224)}
    output_dict = transform(input_dict)
    # Add assertions to check if the brightness and contrast have been modified correctly


def test_TemporalScale():
    transform = TemporalScale((128, 128))
    input_dict = {"video": torch.rand(2, 3, 8, 224, 224)}
    output_dict = transform(input_dict)
    assert output_dict["video"].shape[-2:] == (
        128,
        128,
    ), "TemporalScale did not correctly resize the video."


def test_TemporalJitter():
    transform = TemporalJitter(0.1)
    input_dict = {"video": torch.rand(2, 3, 8, 224, 224)}
    output_dict = transform(input_dict)
    # Add assertions to check if the video has been jittered correctly


# Run the test functions
test_TemporalCrop()
test_TemporalFlip()
test_TemporalRotation()
test_TemporalBrightnessContrast()
test_TemporalScale()
test_TemporalJitter()
