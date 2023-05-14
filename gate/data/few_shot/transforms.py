from typing import Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms.transforms import _setup_size
from torchvision.utils import _log_api_usage_once, save_image
import os
import random

from gate.datasets.tf_hub.few_shot.base import CardinalityType


def channels_first(x):
    return x.transpose([2, 0, 1])


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class SuperClassExistingLabels(torch.nn.Module):
    def __init__(self, num_classes_to_group: Union[int, Tuple[int, int]]):
        super().__init__()
        self.num_classes_to_group = num_classes_to_group
        if isinstance(num_classes_to_group, Tuple):
            (
                self.min_num_classes_to_group,
                self.max_num_classes_to_group,
            ) = num_classes_to_group

    def _group_targets(self, targets, num_classes_to_group):
        targets = torch.tensor(targets)
        class_groupings = targets.unique(sorted=False).chunk(
            len(targets.unique()) // num_classes_to_group
        )

        class_mappings = {
            v_item: k
            for k, v in enumerate(class_groupings)
            for v_item in v.tolist()
        }

        return targets.clone().cpu().apply_(class_mappings.get)

    def forward(self, x):
        if isinstance(self.num_classes_to_group, Tuple):
            num_classes_to_group = torch.randint(
                self.min_num_classes_to_group,
                self.max_num_classes_to_group,
                (1,),
            ).item()
        else:
            num_classes_to_group = self.num_classes_to_group

        new_targets = self._group_targets(x, num_classes_to_group)
        # print(f"Old targets: {x}, New targets: {new_targets}")
        return new_targets


class RandomCropResizeCustom(torch.nn.Module):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions,
    but if non-constant padding is used, the input is expected to have at most
     2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as
             (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided
            this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4
             is provided
            this is the padding for the left, top, right and bottom borders
            respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use
                 a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is
         0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect
         or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with
             fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded
              instead of the last 2

            - reflect: pads with reflection of image without repeating the last
            value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides
              in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value
            on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in
               symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    @staticmethod
    def get_params(
        img: torch.Tensor, output_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger then input image size"
                f" {(h, w)}"
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(
        self,
        size=None,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
    ):
        # if size is None use random size within the bounds of the image size,
        # else use the size provided

        super().__init__()
        _log_api_usage_once(self)
        if size is not None:
            self.size = tuple(
                _setup_size(
                    size,
                    error_msg="Please provide only two dimensions (h, w) for "
                    "size.",
                )
            )
        else:
            self.size = size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

        self.augment = nn.Sequential(
            RandomApply(transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            RandomApply(transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        )

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if img is dict:
            img = img["image"]

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)

        if self.size is None:
            size_h = torch.randint(low=1, high=height, size=(1,)).tolist()
            size_w = torch.randint(low=1, high=width, size=(1,)).tolist()
            size = (size_h[0], size_w[0])

        else:
            size = self.size

        # pad the width if needed
        if self.pad_if_needed and width < size[1]:
            padding = [size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < size[0]:
            padding = [0, size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, size)

        cropped_image = F.crop(img, i, j, h, w)
        resized_image = F.resize(cropped_image, size=(height, width))
        resized_image = self.augment(resized_image)

        if img is dict:
            img["image"] = resized_image
            img["crop_coordinates"] = torch.tensor([i, j, h, w])
            img["cardinality-type"] = CardinalityType.one_to_one
        else:
            img = {
                "image": resized_image,
                "crop_coordinates": torch.tensor([i, j, h, w]),
                "cardinality-type": CardinalityType.one_to_one,
            }

        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self.size},"
            f" padding={self.padding})"
        )


class MultipleRandomCropResizeCustom(RandomCropResizeCustom):
    def __init__(
        self,
        num_augmentations: int,
        min_num_augmentations=-1,
        size=None,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
    ):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.num_augmentations = num_augmentations
        self.min_num_augmentations = min_num_augmentations

    def forward(self, img):
        multiple_img = {
            "image": [],
            "crop_coordinates": [],
            "cardinality-type": CardinalityType.one_to_many,
        }
        if self.min_num_augmentations != -1:
            idx = torch.randint(
                0,
                self.num_augmentations - self.min_num_augmentations,
                size=(1,),
            ).item()
            num_augmentations = list(
                range(self.min_num_augmentations, self.num_augmentations)
            )[idx]
        else:
            num_augmentations = self.num_augmentations
        for _ in range(num_augmentations):
            single_img = super().forward(img=img)
            multiple_img["image"].append(single_img["image"])
            multiple_img["crop_coordinates"].append(
                single_img["crop_coordinates"]
            )

        return multiple_img


class RandomMaskCustom(torch.nn.Module):
    """Mask the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions,
    but if non-constant padding is used, the input is expected to have at most
     2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as
             (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided
            this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4
             is provided
            this is the padding for the left, top, right and bottom borders
            respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use
                 a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is
         0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect
         or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with
             fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded
              instead of the last 2

            - reflect: pads with reflection of image without repeating the last
            value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides
              in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value
            on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in
               symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    @staticmethod
    def get_params(
        img: torch.Tensor, output_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``mask`` for a random mask.

        Args:
            img (PIL Image or Tensor): Image to be masked.
            output_size (tuple): Expected output size of the mask.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``mask`` for random mask.
        """
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                f"Required mask size {(th, tw)} is larger then input image size"
                f" {(h, w)}"
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(
        self,
        size=None,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
    ):
        # if size is None use random size within the bounds of the image size,
        # else use the size provided

        super().__init__()
        _log_api_usage_once(self)
        if size is not None:
            self.size = tuple(
                _setup_size(
                    size,
                    error_msg="Please provide only two dimensions (h, w) for "
                    "size.",
                )
            )
        else:
            self.size = size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if img is dict:
            img = img["image"]

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)

        if self.size is None:
            size_h = torch.randint(low=1, high=height, size=(1,)).tolist()
            size_w = torch.randint(low=1, high=width, size=(1,)).tolist()
            size = (size_h[0], size_w[0])

        else:
            size = self.size

        # pad the width if needed
        if self.pad_if_needed and width < size[1]:
            padding = [size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < size[0]:
            padding = [0, size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, size)

        mask = torch.ones_like(img)
        mask[:, i : i + h, j : j + w] = 0
        masked_image = img * mask + (1 - mask) * torch.rand_like(img)

        # if not os.path.isfile('./test.png'):
        #     save_image(masked_image, 'test.png')
        # else:
        #     if os.path.isfile('./test.png') and not os.path.isfile('./test1.png'):
        #         save_image(masked_image, 'test1.png')
        #     else:
        #         if os.path.isfile('./test.png') and os.path.isfile('./test1.png') and not os.path.isfile('./test2.png'):
        #             save_image(masked_image, 'test2.png')

        if img is dict:
            img["image"] = masked_image
            img["crop_coordinates"] = torch.tensor([i, j, h, w])
            img["cardinality-type"] = CardinalityType.one_to_one
        else:
            img = {
                "image": masked_image,
                "crop_coordinates": torch.tensor([i, j, h, w]),
                "cardinality-type": CardinalityType.one_to_one,
            }

        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self.size},"
            f" padding={self.padding})"
        )
