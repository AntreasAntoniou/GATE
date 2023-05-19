import PIL
import torch
import torchvision.transforms as T


def image_dim_reshape(x):
    if len(x.shape) == 5:
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])

    return x


def apply_preprocessing_transforms(transforms, x):
    input_shape = None
    if isinstance(x, PIL.Image.Image):
        x = T.ToTensor()(x)

    if isinstance(x, torch.Tensor):
        input_shape = x.shape
        x = image_dim_reshape(x)

    if transforms is not None:
        x = transforms(x)

    if input_shape is not None:
        x = x.reshape(input_shape)

    return x
