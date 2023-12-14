import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor

from gate.models.task_adapters.utils import reinit

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
        x = T.ToPILImage()(x)

    if isinstance(x, torch.Tensor) and modality == Modality.image:
        input_shape = x.shape
        is_5d_tensor = len(x.shape) == 5
        x = image_dim_reshape(x)

    if transforms is not None:
        x = transforms(x)

    if (
        input_shape is not None
        and isinstance(x, torch.Tensor)
        and is_5d_tensor
    ):
        x = x.view(input_shape[0], input_shape[1], *x.shape[1:])

    return x


def interpolate_position_encoding(
    pos_embed: Tensor,
    x: Tensor,
    w: int,
    h: int,
    patch_size: int,
    batch_size: int,
    class_token_idx: Optional[int] = None,
) -> Tensor:
    """
    Interpolate the position encoding based on the input tensor dimensions.

    Args:
        pos_embed (torch.Tensor): Position embedding tensor.
            Shape: (B, N, D) or (B, N+1, D), depending on the class token.
        x (torch.Tensor): Input tensor.
            Shape: Either (B, N, D) or (B, C, H, W).
        w (int): Width of the input.
        h (int): Height of the input.
        patch_size (int): Patch size used in the patch embedding module.
        batch_size (int): The batch size, required to process the data.
        class_token_idx (Optional[int], optional): Index of the class token,
            if present. Defaults to None.

    Returns:
        torch.Tensor: Interpolated position encoding tensor.
            Shape: (B, npatch+1, D) or (B, npatch, D), depending on the class token.
    """

    N = pos_embed.shape[1] - (1 if class_token_idx is not None else 0)
    npatch = (w // patch_size) * (h // patch_size)

    if npatch == N:
        return pos_embed

    # Separate class and patch position embeddings
    if class_token_idx is not None:
        class_pos_embed = pos_embed[:, class_token_idx]
        patch_pos_embed = torch.cat(
            (
                pos_embed[:, :class_token_idx],
                pos_embed[:, class_token_idx + 1 :],
            ),
            dim=1,
        )
    else:
        class_pos_embed = None
        patch_pos_embed = pos_embed

    dim = x.shape[-1]
    w0, h0 = w // patch_size, h // patch_size

    # Add a small number to avoid floating point errors ⚠️
    w0, h0 = w0 + 0.1, h0 + 0.1

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(
        batch_size, -1, dim
    )

    # Add the class position embedding back if it was present
    if class_token_idx is not None:
        if class_pos_embed is not None:
            return torch.cat(
                (class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1
            )

    return patch_pos_embed


class VisionTextGATEAdapter(ABC):
    @abstractmethod
    def __init__(
        self,
    ):
        pass

    def init_weights(self):
        reinit(self)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if image is None and text is None and video is None:
            raise ValueError(
                f"Must provide at least one input modality"
                f"to {self.__class__.__name__}"
            )
        output_dict = defaultdict(dict)

        if image is not None:
            output_dict["image"] = self.vision_model(image=image)
            output_dict["image"]["features"] = self.visual_projection(
                output_dict["image"]["features"]
            )

        if video is not None:
            if len(video.shape) == 5:
                b, s, c, h, w = video.shape
                output_dict["video"] = self.vision_model.forward(
                    image=video.view(b * s, c, h, w)
                )
                for k, v in output_dict["video"].items():
                    if v is not None:
                        if isinstance(v, list) or isinstance(v, tuple):
                            v = torch.stack(v, dim=2)

                        output_dict["video"][k] = v.view(b, s, *v.shape[1:])
            else:
                output_dict["video"] = self.vision_model.forward(image=video)

            output_dict["video"]["features"] = self.visual_projection(
                output_dict["video"]["features"]
            )

        if text is not None:
            output_dict["text"] = self.text_model(text=text)
            output_dict["text"]["features"] = self.text_projection(
                output_dict["text"]["features"]
            )

        return output_dict

    def get_transforms(self, image_size: int = 224):
        def image_transforms(x):
            return self.preprocessor(
                images=T.Resize(size=(image_size, image_size), antialias=True)(
                    x
                ),
                do_resize=False,
                do_center_crop=False,
                return_tensors="pt",
            ).pixel_values.squeeze(0)

        def text_transforms(x):
            return self.text_transforms.apply_transform(x)

        def image_transforms_process_multi_type(x):
            if isinstance(x, List):
                return [
                    apply_preprocessing_transforms(
                        x=item,
                        transforms=image_transforms,
                        modality=Modality.image,
                    )
                    for item in x
                ]
            else:
                return apply_preprocessing_transforms(
                    x=x, transforms=image_transforms, modality=Modality.image
                )

        def text_transforms_process_multi_type(x):
            return apply_preprocessing_transforms(
                x=x, transforms=text_transforms, modality=Modality.text
            )

        def video_transforms_process_multi_type(x):
            return torch.stack(
                [image_transforms_process_multi_type(item) for item in x],
                dim=0,
            )

        return {
            "image": lambda x: image_transforms_process_multi_type(x),
            "text": lambda x: text_transforms_process_multi_type(x),
            "video": lambda x: video_transforms_process_multi_type(x),
        }


def forward_dict(
    self, image: Optional[torch.Tensor] = None, text: Optional[str] = None
):
    if image is not None and text is not None:
        x = [image, text]
    elif image is not None:
        x = image
    elif text is not None:
        x = text
    else:
        raise ValueError(
            f"Must provide at least one input modality"
            f"to {self.__class__.__name__}"
        )

    output = self.legacy_forward(
        x, return_dict=False, output_hidden_states=True
    )
    (last_hidden_state, pooled_output, encoder_outputs) = output
    encoder_outputs = [f for f in encoder_outputs]

    return {
        "features": pooled_output,
        "raw_features": last_hidden_state,
        "per_layer_raw_features": encoder_outputs,
    }


class TextProcessor:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def text_transforms(self, x: Union[List[str], List[List[str]]]):
        if isinstance(x[0], list):
            x = [item for sublist in x for item in sublist]
        return self.preprocessor(
            text=x, return_tensors="pt", padding=True, truncation=True
        ).input_ids.squeeze(0)

    def apply_transform(self, text: Union[List[str], List[List[str]]]):
        if not all(
            isinstance(i, list) for i in text
        ):  # if text is list of strings
            text = [text]
        transformed_text = self.text_transforms(text)
        return transformed_text


class GATEncoder(ABC, nn.Module):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def image_shape(self) -> Tuple[int, int]:
        pass

    @property
    @abstractmethod
    def num_in_features_image(self):
        pass

    @property
    @abstractmethod
    def num_in_features_text(self):
        pass

    @property
    @abstractmethod
    def num_in_features_video(self):
        pass

    @property
    @abstractmethod
    def num_raw_features_image(self):
        pass

    @property
    @abstractmethod
    def num_raw_features_text(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def get_transforms(self):
        pass
