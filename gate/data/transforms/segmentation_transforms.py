# Importing required modules
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
    2. Random contrast
    3. Random saturation
    4. Random hue

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
            img = T.ToTensor()(img)

            if img.shape[0] == 1:
                transform = T.Compose(
                    [
                        T.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                )
                img = transform(img)

            return img
        else:
            raise TypeError(f"Unsupported image type {type(img)}")

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
                    cv2.COLOR_RGB2HSV,
                )
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

    def _convert_to_rgb(self, img):
        """Convert image back to BGR."""
        return (
            torch.tensor(
                cv2.cvtColor(
                    (img * 255).byte().cpu().numpy().transpose(1, 2, 0),
                    cv2.COLOR_HSV2RGB,
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

        img = self._convert_to_rgb(img_hsv)

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
        annotation = inputs[self.label_label]

        if isinstance(image, dict):
            image = image["bytes"]
            # Create a BytesIO object and read the bytes into it
            image = io.BytesIO(image)
            # Use PIL to open the image from the BytesIO object
            image = Image.open(image)

        if isinstance(annotation, dict):
            annotation = annotation["bytes"]
            # Create a BytesIO object and read the bytes into it
            annotation = io.BytesIO(annotation)
            # Use PIL to open the image from the BytesIO object
            annotation = Image.open(annotation)

        image = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )(image)

        annotation = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )(annotation)

        return {
            "image": image,
            "labels": annotation,
        }


def is_grayscale(image):
    """
    Check if an image is grayscale.

    Parameters:
        image (PIL.Image.Image | np.ndarray | torch.Tensor): The image to check.

    Returns:
        bool: True if the image is grayscale, False otherwise.

    Raises:
        TypeError: If the input image type is not supported.
    """
    if isinstance(image, Image.Image):
        return image.mode == "L" or image.mode == "1" or image.mode == "I;16"
    elif isinstance(image, np.ndarray):
        return len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[2] == 1
        )
    elif torch.is_tensor(image):
        return len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[0] == 1
        )
    else:
        raise TypeError(
            "Input type not supported. Expected one of [PIL.Image, np.ndarray, torch.Tensor]."
        )


def grayscale_to_rgb(image: Union[Image.Image, np.ndarray, torch.Tensor]):
    """
    Convert a grayscale image to RGB.

    Parameters:
        image (PIL.Image.Image | np.ndarray | torch.Tensor): The grayscale image to convert.

    Returns:
        PIL.Image.Image | np.ndarray | torch.Tensor: The converted RGB image.

    Raises:
        TypeError: If the input image type is not supported.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3:
            return np.concatenate([image] * 3, axis=2)
    elif torch.is_tensor(image):
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        return image.repeat((3, 1, 1))
    else:
        raise TypeError(
            "Input type not supported. Expected one of [PIL.Image, np.ndarray, torch.Tensor]."
        )


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

        if is_grayscale(image):
            image = grayscale_to_rgb(image)

        annotation = inputs["labels"]

        if self.crop_size is not None:
            image, annotation = self.crop_transform(image, annotation)

        if self.flip_probability is not None:
            image, annotation = self.random_flip(image, annotation)

        if self.photo_metric_distortion is not None:
            image = self.photo_metric_distortion(image)

        annotation = np.array(annotation)
        annotation = torch.from_numpy(annotation)
        if annotation.shape[2] == 3:
            annotation = annotation[:, :, 0].unsqueeze(2)

        if len(annotation.shape) == 2:
            annotation = annotation.unsqueeze(0)
        elif len(annotation.shape) == 3:
            annotation = annotation.permute(2, 0, 1)
        else:
            raise ValueError("Unsupported annotation shape")

        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        image = T.Resize(
            (self.input_size[0], self.input_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )(image)

        annotation = T.Resize(
            (self.target_size[0], self.target_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )(annotation)

        return {
            "image": image,
            "labels": annotation.long(),
        }


class MedicalPhotoMetricDistortion:
    """Apply photometric distortion to a medical image."""

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5)):
        self.brightness_delta = brightness_delta / 255.0
        self.contrast_lower, self.contrast_upper = contrast_range

    def _apply_brightness(self, img):
        delta = random.uniform(-self.brightness_delta, self.brightness_delta)
        return img + delta

    def _apply_contrast(self, img):
        alpha = random.uniform(self.contrast_lower, self.contrast_upper)
        return img * alpha

    def __call__(self, img):
        if random.randint(0, 1):
            img = self._apply_brightness(img)
        if random.randint(0, 1):
            img = self._apply_contrast(img)
        return img.clamp(0, 1)


@dataclass
class PhotometricParams:
    brightness_delta: int = 32
    contrast_range: tuple = (0.5, 1.5)


class MedicalImageSegmentationTransforms:
    """
    Apply a series of data augmentation techniques for medical image segmentation.
    """

    def __init__(self, photometric_params: Optional[PhotometricParams] = None):
        self.photometric_transform = (
            MedicalPhotoMetricDistortion()
            if photometric_params is None
            else MedicalPhotoMetricDistortion(**photometric_params.__dict__)
        )

    def _apply_random_rotation(self, img, mask):
        """Apply random rotation in 90-degree increments."""
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle), TF.rotate(mask, angle)

    def _apply_random_flip(self, img, mask):
        """Apply random horizontal flip."""
        if random.random() > 0.5:
            return img.flip(-1), mask.flip(-1)
        return img, mask

    def __call__(self, img, mask):
        """
        Apply the data augmentation techniques.
        """
        # Validate input shape
        if img.shape != mask.shape or len(img.shape) != 4:
            raise ValueError(
                "Input image and mask must have the same shape (num_scan, num_slices, height, width)"
            )

        # Apply random rotation
        img, mask = self._apply_random_rotation(img, mask)

        # Apply random horizontal flip
        img, mask = self._apply_random_flip(img, mask)

        # Apply photometric distortion
        img = self.photometric_transform(img)

        return img, mask
