# Importing required modules
import random
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class DualImageRandomFlip:
    def __init__(self, p=0.5):
        """
        :param p: Probability of applying the flip. Default is 0.5.
        """
        self.p = p

    def __call__(self, img1, img2):
        """
        :param img1: First image to be flipped.
        :param img2: Second image to be flipped.
        :return: Tuple of flipped or original images.
        """
        # Check if img1 and img2 are of the same type and handle accordingly
        if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
            h1, w1 = img1.shape[-2:]
            h2, w2 = img2.shape[-2:]
            if (h1, w1) != (h2, w2):
                raise ValueError("Input images must have the same dimensions!")

            # Perform random flip with probability p
            if random.random() < self.p:
                img1 = torch.flip(img1, [-1])
                img2 = torch.flip(img2, [-1])

        elif isinstance(img1, Image.Image) and isinstance(img2, Image.Image):
            w1, h1 = img1.size
            w2, h2 = img2.size
            if (h1, w1) != (h2, w2):
                raise ValueError("Input images must have the same dimensions!")

            # Perform random flip with probability p
            if random.random() < self.p:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
                img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

        else:
            raise TypeError(
                "Unsupported type for img1 and/or img2, or the types do not match"
            )

        return img1, img2


class DualImageRandomCrop:
    def __init__(self, output_size):
        """
        :param output_size: Desired output size of the crop. Either int or Tuple.
        """
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, img1, img2):
        """
        :param img1: First image to be cropped.
        :param img2: Second image to be cropped.
        :return: Tuple of cropped images.
        """
        if isinstance(img1, torch.Tensor):
            h1, w1 = img1.shape[-2:]
        elif isinstance(img1, Image.Image):
            w1, h1 = img1.size
        else:
            raise TypeError("Unsupported type for img1")

        if isinstance(img2, torch.Tensor):
            h2, w2 = img2.shape[-2:]
        elif isinstance(img2, Image.Image):
            w2, h2 = img2.size
        else:
            raise TypeError("Unsupported type for img2")

        if (h1, w1) != (h2, w2):
            raise ValueError("Input images must have the same dimensions!")

        new_h, new_w = self.output_size

        # Generate random coordinates for the top left corner of the crop
        top = random.randint(0, h1 - new_h)
        left = random.randint(0, w1 - new_w)

        # Crop the images using the generated coordinates
        if isinstance(img1, torch.Tensor):
            img1 = img1[..., top : top + new_h, left : left + new_w]
        else:
            img1 = img1.crop((left, top, left + new_w, top + new_h))

        if isinstance(img2, torch.Tensor):
            img2 = img2[..., top : top + new_h, left : left + new_w]
        else:
            img2 = img2.crop((left, top, left + new_w, top + new_h))

        return img1, img2


class PhotoMetricDistortion:
    """
    Apply photometric distortion to an image.
    The image should be either a PyTorch tensor, a NumPy array, or a PIL Image.

    The following distortions are applied:
    1. Random brightness
    2. Random contrast (mode 0)
    3. Convert color from BGR to HSV
    4. Random saturation
    5. Random hue
    6. Convert color from HSV to BGR
    7. Random contrast (mode 1)

    Parameters:
    - brightness_delta (int): Delta value for brightness adjustment.
    - contrast_range (tuple): Range for random contrast adjustment.
    - saturation_range (tuple): Range for random saturation adjustment.
    - hue_delta (int): Delta value for hue adjustment.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta / 255.0
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta / 360.0

    def _convert_to_tensor(self, img):
        """Convert the input image to a PyTorch tensor."""
        if isinstance(img, torch.Tensor):
            return img.clone()
        elif isinstance(img, np.ndarray):
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        elif isinstance(img, Image.Image):
            return torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        else:
            raise TypeError("Unsupported image type")

    def _apply_brightness(self, img):
        """Apply random brightness adjustment."""
        delta = random.uniform(-self.brightness_delta, self.brightness_delta)
        return img + delta

    def _apply_contrast(self, img):
        """Apply random contrast adjustment."""
        alpha = random.uniform(self.contrast_lower, self.contrast_upper)
        return img * alpha

    def _convert_to_hsv(self, img):
        """Convert image to HSV."""
        return (
            torch.tensor(
                cv2.cvtColor(
                    (img * 255).byte().cpu().numpy().transpose(1, 2, 0),
                    cv2.COLOR_BGR2HSV,
                )
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

    def _convert_to_bgr(self, img):
        """Convert image back to BGR."""
        return (
            torch.tensor(
                cv2.cvtColor(
                    (img * 255).byte().cpu().numpy().transpose(1, 2, 0),
                    cv2.COLOR_HSV2BGR,
                )
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

    def _apply_saturation(self, img):
        """Apply random saturation adjustment."""
        img[1] *= random.uniform(self.saturation_lower, self.saturation_upper)
        return img

    def _apply_hue(self, img):
        """Apply random hue adjustment."""
        img[0] += random.uniform(-self.hue_delta, self.hue_delta)
        img[0][img[0] > 1] -= 1
        img[0][img[0] < 0] += 1
        return img

    def __call__(self, img):
        """
        Apply the photometric distortion.

        Parameters:
        - img: Input image, can be a PyTorch tensor, a NumPy array, or a PIL Image.

        Returns:
        - Transformed image as a PyTorch tensor.
        """
        img = self._convert_to_tensor(img)

        if random.randint(0, 1):
            img = self._apply_brightness(img)
        if random.randint(0, 1):
            img = self._apply_contrast(img)

        img_hsv = self._convert_to_hsv(img)

        if random.randint(0, 1):
            img_hsv = self._apply_saturation(img_hsv)
        if random.randint(0, 1):
            img_hsv = self._apply_hue(img_hsv)

        img = self._convert_to_bgr(img_hsv)

        if random.randint(0, 1):
            img = self._apply_contrast(img)

        img = img.clamp(0, 1)
        return img


class KeySelectorTransforms:
    def __init__(
        self,
        initial_size: Union[int, List[int]] = 1024,
        image_label: str = "image",
        label_label: str = "annotation",
    ):
        self.initial_size = (
            initial_size
            if isinstance(initial_size, tuple)
            or isinstance(initial_size, list)
            else (initial_size, initial_size)
        )
        self.image_label = image_label
        self.label_label = label_label

    def __call__(self, inputs: Dict):
        image = inputs[self.image_label]
        image = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(image)

        annotation = inputs[self.label_label]
        annotation = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(annotation)

        return {
            "image": image,
            "labels": annotation,
        }


class BaseDatasetTransforms:
    def __init__(
        self,
        input_size: Union[int, List[int]],
        target_size: Union[int, List[int]],
        crop_size: Optional[Union[int, List[int]]] = None,
        flip_probability: Optional[float] = None,
        use_photo_metric_distortion: bool = True,
        brightness_delta: int = 32,
        contrast_range: tuple = (0.5, 1.5),
        saturation_range: tuple = (0.5, 1.5),
        hue_delta: int = 18,
    ):
        self.input_size = (
            input_size
            if isinstance(input_size, tuple) or isinstance(input_size, list)
            else (input_size, input_size)
        )
        self.target_size = (
            target_size
            if isinstance(target_size, tuple) or isinstance(target_size, list)
            else (target_size, target_size)
        )
        if crop_size is not None:
            self.crop_size = (
                crop_size
                if isinstance(crop_size, list) or isinstance(crop_size, tuple)
                else [crop_size, crop_size]
            )
            self.crop_transform = DualImageRandomCrop(self.crop_size)
        else:
            self.crop_size = None

        if flip_probability is not None:
            self.flip_probability = flip_probability
            self.random_flip = DualImageRandomFlip(p=flip_probability)
        else:
            self.flip_probability = None

        if use_photo_metric_distortion:
            self.photo_metric_distortion = PhotoMetricDistortion(
                brightness_delta=brightness_delta,
                contrast_range=contrast_range,
                saturation_range=saturation_range,
                hue_delta=hue_delta,
            )
        else:
            self.photo_metric_distortion = None

    def __call__(self, inputs: Dict):
        image = inputs["image"]
        annotation = inputs["labels"]

        if self.crop_size is not None:
            image, annotation = self.crop_transform(image, annotation)

        if self.flip_probability is not None:
            image, annotation = self.random_flip(image, annotation)

        if self.photo_metric_distortion is not None:
            image = self.photo_metric_distortion(image)

        annotation = np.array(annotation)
        annotation = torch.from_numpy(annotation)
        annotation = annotation.permute(2, 0, 1)[0].unsqueeze(0)

        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        image = T.Resize(
            (self.input_size[0], self.input_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(image)

        annotation = T.Resize(
            (self.target_size[0], self.target_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(annotation)

        return {
            "image": image,
            "labels": annotation.long(),
        }
