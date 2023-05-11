from torchvision import transforms


def pad_image(image, target_size=224):
    w, h = image.size
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    padding = (pad_w, pad_h, target_size - w - pad_w, target_size - h - pad_h)
    return transforms.functional.pad(image, padding, fill=0)
