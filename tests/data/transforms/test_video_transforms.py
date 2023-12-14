import torch

from gate.data.transforms.video import (BaseVideoTransform,
                                        TemporalBrightnessContrast,
                                        TemporalCrop, TemporalFlip,
                                        TemporalJitter, TemporalRotation,
                                        TemporalScale, TrainVideoTransform)


def test_TemporalCrop():
    transform = TemporalCrop((200, 200))
    input_dict = {"video": torch.rand(2, 8, 3, 224, 224)}
    output_dict = transform(input_dict)
    assert output_dict["video"].shape[-2:] == (
        200,
        200,
    ), "TemporalCrop did not correctly crop the video."


def test_TemporalFlip():
    transform = TemporalFlip()
    input_dict = {"video": torch.rand(2, 8, 3, 224, 224)}
    transform(input_dict)


def test_TemporalRotation():
    transform = TemporalRotation()
    input_dict = {"video": torch.rand(2, 8, 3, 224, 224)}
    transform(input_dict)
    # Add assertions to check if the video has been rotated correctly


def test_TemporalBrightnessContrast():
    transform = TemporalBrightnessContrast()
    input_dict = {"video": torch.rand(2, 8, 3, 224, 224)}
    transform(input_dict)
    # Add assertions to check if the brightness and contrast have been modified correctly


def test_TemporalScale():
    transform = TemporalScale((128, 128))
    input_dict = {"video": torch.rand(2, 8, 3, 224, 224)}
    output_dict = transform(input_dict)
    assert output_dict["video"].shape[-2:] == (
        128,
        128,
    ), "TemporalScale did not correctly resize the video."


def test_TemporalJitter():
    transform = TemporalJitter(0.1)
    input_dict = {"video": torch.rand(2, 8, 3, 224, 224)}
    transform(input_dict)
    # Add assertions to check if the video has been jittered correctly


def test_BaseVideoTransform():
    transform = BaseVideoTransform(scale_factor=(224, 224))
    input_dict = {"video": torch.rand(2, 8, 3, 640, 480)}
    output_dict = transform(input_dict)
    assert output_dict["video"].shape[-2:] == (
        224,
        224,
    ), "BaseVideoTransform did not correctly scale the video."


def test_TrainVideoTransform():
    transform = TrainVideoTransform(
        scale_factor=(448, 448),
        crop_size=(224, 224),
        flip_prob=0.5,
        rotation_angles=[0, 90, 180, 270],
        brightness=0.2,
        contrast=0.2,
        jitter_strength=0.1,
    )
    input_dict = {"video": torch.rand(2, 8, 3, 640, 480)}
    output_dict = transform(input_dict)

    assert output_dict["video"].shape[-2:] == (
        224,
        224,
    ), "TrainVideoTransform did not correctly transform the video."
    # Additional assertions can be added to verify other transformations
