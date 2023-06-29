import random

import torch
from PIL import Image
from torchvision import transforms


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
