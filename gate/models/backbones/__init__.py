from dataclasses import dataclass
import PIL
import torch
import torchvision.transforms as T

single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))


def image_dim_reshape(x):
    if len(x.shape) == 5:
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])

    return x


@dataclass
class Modality:
    image: str = "image"
    text: str = "text"
    audio: str = "audio"
    video: str = "video"


def apply_preprocessing_transforms(transforms, x, modality=Modality.image):
    input_shape = None
    is_5d_tensor = False
    if isinstance(x, PIL.Image.Image) and modality == Modality.image:
        x = T.ToTensor()(x)
        if x.shape[0] == 1:
            x = single_to_three_channel(x)

    if isinstance(x, torch.Tensor) and modality == Modality.image:
        input_shape = x.shape
        x = image_dim_reshape(x)
        is_5d_tensor = len(x.shape) == 5

    if transforms is not None:
        x = transforms(x)

    if (
        input_shape is not None
        and isinstance(x, torch.Tensor)
        and is_5d_tensor
    ):
        x = x.view(input_shape[0], input_shape[1], *x.shape[1:])

    return x
