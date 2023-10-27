from typing import List, Tuple, Union

import numpy as np
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def pad_image(image, target_size: Union[int, List, Tuple] = 224):
    from_PIL = isinstance(image, PIL.Image.Image)
    if from_PIL:
        image = T.ToTensor()(image)

    w, h = image.shape[-2:]
    target_w = target_size if isinstance(target_size, int) else target_size[0]
    pad_w = (target_w - w) // 2
    target_h = target_size if isinstance(target_size, int) else target_size[1]
    pad_h = (target_h - h) // 2
    padding = (pad_w, pad_h, target_w - w - pad_w, target_h - h - pad_h)

    if from_PIL:
        image = T.ToPILImage()(image)
    return TF.pad(image, padding, fill=0)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert a PIL Image of any channel format (RGB, RGBA, L, etc.) to RGB format.

    Parameters:
    - image (PIL.Image.Image): Input image in any channel format.

    Returns:
    - PIL.Image.Image: Output image in RGB format.

    😎 Efficiently handles different channel styles to ensure RGB output.
    """
    # Check if the image is already in RGB format
    if image.mode == "RGB":
        return image

    # If the image is grayscale (L), convert it to RGB
    elif image.mode == "L":
        return image.convert("RGB")

    # If the image has an alpha channel (RGBA), remove the alpha channel
    elif image.mode == "RGBA":
        # Create an RGB image with the same size as the input image
        rgb_image = Image.new("RGB", image.size)
        # Paste the RGBA image into the RGB image, effectively dropping the alpha channel
        rgb_image.paste(
            image, mask=image.split()[3]
        )  # 3 is the index of the alpha channel
        return rgb_image

    # If the image mode is not recognized, raise an error
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")
