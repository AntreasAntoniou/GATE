import random

import torch
from torchvision.transforms import Resize


class TemporalCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, input_dict):
        video = input_dict["video"]
        # Assume video shape: (batch, channel, time, height, width)
        if len(video.shape) == 5:
            B, S, C, H, W = video.shape
            top = random.randint(0, H - self.crop_size[0])
            left = random.randint(0, W - self.crop_size[1])
            input_dict["video"] = video[
                :,
                :,
                :,
                top : top + self.crop_size[0],
                left : left + self.crop_size[1],
            ]
        else:
            _, _, H, W = video.shape
            top = random.randint(0, H - self.crop_size[0])
            left = random.randint(0, W - self.crop_size[1])
            input_dict["video"] = video[
                :,
                :,
                top : top + self.crop_size[0],
                left : left + self.crop_size[1],
            ]
        return input_dict


class TemporalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, input_dict):
        video = input_dict["video"]
        if random.random() < self.flip_prob:
            input_dict["video"] = video.flip(-1)
        return input_dict


class TemporalRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, input_dict):
        video = input_dict["video"]
        angle = random.choice(self.angles)
        if angle != 0:
            input_dict["video"] = torch.rot90(
                video, k=angle // 90, dims=[-2, -1]
            )
        return input_dict


class TemporalBrightnessContrast:
    def __init__(self, max_brightness=0.2, max_contrast=0.2):
        self.max_brightness = max_brightness
        self.max_contrast = max_contrast

    def __call__(self, input_dict):
        video = input_dict["video"]
        # Randomly choose brightness and contrast factors
        brightness = random.uniform(0, self.max_brightness)
        contrast = random.uniform(1, 1 + self.max_contrast)

        # Apply brightness and contrast
        x_transformed = contrast * (video - 0.5) + 0.5 + brightness

        # Clip values to be in [0, 1]
        x_transformed = torch.clamp(x_transformed, 0, 1)

        input_dict["video"] = x_transformed

        return input_dict


class TemporalScale:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.resizer = Resize(scale_factor)

    def __call__(self, input_dict):
        video = input_dict["video"]
        # Rescale all the frames
        # Assume video shape: (batch, channel, time, height, width)
        if len(video.shape) == 5:
            b, s, c, h, w = video.shape
            video_resized = self.resizer(video.view(b * s, c, h, w))
            input_dict["video"] = video_resized.view(
                b, s, c, *video_resized.shape[-2:]
            )
        else:
            t, c, h, w = video.shape
            video_resized = self.resizer(video)
            input_dict["video"] = video_resized
        return input_dict


class TemporalJitter:
    def __init__(self, jitter_strength):
        self.jitter_strength = jitter_strength

    def __call__(self, input_dict):
        video = input_dict["video"]
        noise = torch.randn_like(video) * self.jitter_strength
        input_dict["video"] = torch.clamp(video + noise, 0, 1)
        return input_dict


class BaseVideoTransform:
    def __init__(self, scale_factor=(224, 224)):
        self.scale = TemporalScale(scale_factor)

    def __call__(self, input_dict):
        return self.scale(input_dict)


class TrainVideoTransform:
    def __init__(
        self,
        scale_factor=(448, 448),
        crop_size=(224, 224),
        flip_prob=0.5,
        rotation_angles=[0, 90, 180, 270],
        brightness=0.2,
        contrast=0.2,
        jitter_strength=0.1,
    ):
        self.scale = TemporalScale(scale_factor)
        self.crop = TemporalCrop(crop_size)
        self.flip = TemporalFlip(flip_prob)
        self.rotation = TemporalRotation(rotation_angles)
        self.brightness_contrast = TemporalBrightnessContrast(
            brightness, contrast
        )
        self.jitter = TemporalJitter(jitter_strength)

    def __call__(self, input_dict):
        # print("Before scale", input_dict["video"].shape)
        input_dict = self.scale(input_dict)
        # print("After scale", input_dict["video"].shape)
        input_dict = self.crop(input_dict)
        # print(f"After crop {input_dict['video'].shape}")
        input_dict = self.flip(input_dict)
        # # print(f"After flip {input_dict['video'].shape}")
        input_dict = self.rotation(input_dict)
        # # print(f"After rotation {input_dict['video'].shape}")
        input_dict = self.brightness_contrast(input_dict)
        # # print(f"After brightness contrast {input_dict['video'].shape}")
        input_dict = self.jitter(input_dict)
        # print(f"After jitter {input_dict['video'].shape}")

        return input_dict
