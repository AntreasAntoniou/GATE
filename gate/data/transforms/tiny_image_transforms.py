from typing import List, Tuple, Union

import PIL
from torchvision import transforms


def pad_image(image, target_size: Union[int, List, Tuple] = 224):
    from_PIL = isinstance(image, PIL.Image.Image)
    if from_PIL:
        image = transforms.ToTensor()(image)

    w, h = image.shape[-2:]
    target_w = target_size if isinstance(target_size, int) else target_size[0]
    pad_w = (target_w - w) // 2
    target_h = target_size if isinstance(target_size, int) else target_size[1]
    pad_h = (target_h - h) // 2
    padding = (pad_w, pad_h, target_w - w - pad_w, target_h - h - pad_h)

    if from_PIL:
        image = transforms.ToPILImage()(image)
    return transforms.functional.pad(image, padding, fill=0)
