import numpy as np
import torch
from PIL import Image

from gate.data.transforms.segmentation import grayscale_to_rgb, is_grayscale


def test_is_grayscale():
    # Test with PIL Image
    pil_image_gray = Image.new("L", (4, 4))
    assert is_grayscale(pil_image_gray) == True

    pil_image_rgb = Image.new("RGB", (4, 4))
    assert is_grayscale(pil_image_rgb) == False

    # Test with NumPy array
    numpy_image_gray = np.random.rand(4, 4)
    assert is_grayscale(numpy_image_gray) == True

    numpy_image_rgb = np.random.rand(4, 4, 3)
    assert is_grayscale(numpy_image_rgb) == False

    # Test with PyTorch tensor
    torch_image_gray = torch.rand((1, 4, 4))
    assert is_grayscale(torch_image_gray) == True

    torch_image_rgb = torch.rand((3, 4, 4))
    assert is_grayscale(torch_image_rgb) == False


def test_grayscale_to_rgb():
    # Test with PIL Image
    pil_image_gray = Image.new("L", (4, 4))
    pil_image_rgb = grayscale_to_rgb(pil_image_gray)
    assert pil_image_rgb.mode == "RGB"

    # Test with NumPy array
    numpy_image_gray = np.random.rand(4, 4)
    numpy_image_rgb = grayscale_to_rgb(numpy_image_gray)
    assert numpy_image_rgb.shape == (4, 4, 3)

    # Test with PyTorch tensor
    torch_image_gray = torch.rand((1, 4, 4))
    torch_image_rgb = grayscale_to_rgb(torch_image_gray)
    assert torch_image_rgb.shape == (3, 4, 4)
