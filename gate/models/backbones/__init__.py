import gc
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor
from transformers.models.clip.modeling_clip import CLIPOutput

from gate.models.core import simple_init

single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))


def reinit(input_module: nn.Module):
    for name, module in input_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


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

    def process_images(self, image: torch.Tensor) -> torch.Tensor:
        if image is None:
            raise ValueError("Image cannot be None.")

        output = None
        batch_size = image.shape[0]
        # print(f"batch_size: {batch_size}")
        try:
            return self.vision_model(image=image)

        except torch.cuda.OutOfMemoryError as e:
            # clear cache and try again
            self.zero_grad()
            self.vision_model.zero_grad()
            torch.cuda.empty_cache()
            while batch_size >= 1:
                print(f"batch_size: {batch_size}")
                batches = torch.split(image, batch_size)
                print("Before forward pass:")
                outputs = []
                try:
                    print("Before Before forward pass:")
                    for batch in batches:
                        print(f"batch shape: {batch.shape}")
                        output = self.vision_model(image=batch)
                        outputs.append(output.cpu())
                        # Print memory usage after forward pass
                        output = output.detach().cpu()
                        del output
                        batch = batch.cpu()
                        torch.cuda.empty_cache()
                        gc.collect()

                    if isinstance(outputs[0], dict):
                        output_dict = {}
                        for k, v in outputs[0].items():
                            output_dict[k] = torch.cat(
                                [output[k] for output in outputs], dim=0
                            )
                        return output_dict
                    elif isinstance(outputs[0], torch.Tensor):
                        return torch.cat(outputs, dim=0)

                except torch.cuda.OutOfMemoryError as inner_e:
                    # clear cache and try again
                    self.zero_grad()
                    self.vision_model.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                    batch_size = batch_size // 2
                    continue
                except Exception as inner_e:
                    raise inner_e

            raise RuntimeError(f"Even with batch size 1, still failed.")
        except Exception as e:
            raise e

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
            output_dict["image"] = self.process_images(image=image)
            output_dict["image"]["features"] = self.visual_projection(
                output_dict["image"]["features"]
            )

        if video is not None:
            if len(video.shape) == 5:
                b, s, c, h, w = video.shape
                output_dict["video"] = self.process_images(
                    image=video.view(b * s, c, h, w)
                )
                for k, v in output_dict["video"].items():
                    if v is not None:
                        if isinstance(v, list) or isinstance(v, tuple):
                            v = torch.stack(v, dim=2)

                        output_dict["video"][k] = v.view(b, s, *v.shape[1:])
            else:
                output_dict["video"] = self.process_images(image=video)

            output_dict["video"]["features"] = self.visual_projection(
                output_dict["video"]["features"]
            )

        if text is not None:
            output_dict["text"] = self.text_model(text=text)
            output_dict["text"]["features"] = self.text_projection(
                output_dict["text"]["features"]
            )

        return output_dict

    def get_transforms(self, image_size: Optional[int] = 224):
        if image_size is None:
            image_size = 224

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


class GATEImageEncoder(ABC, nn.Module):
    @property
    @abstractmethod
    def projection_layer(self):
        pass

    @property
    @abstractmethod
    def num_projection_features(self):
        pass

    @property
    @abstractmethod
    def num_features(self):
        pass

    @property
    @abstractmethod
    def num_raw_features(self):
        pass

    @property
    @abstractmethod
    def image_shape(self):
        pass

    @abstractmethod
    def forward(self, args, **kwargs):
        pass

    @abstractmethod
    def transforms(self, x):
        pass


class GATETextEncoder(ABC, nn.Module):
    @property
    @abstractmethod
    def projection_layer(self):
        pass

    @property
    @abstractmethod
    def num_projection_features(self):
        pass

    @property
    @abstractmethod
    def num_features(self):
        pass

    @property
    @abstractmethod
    def num_raw_features(self):
        pass

    @abstractmethod
    def forward(self, args, **kwargs):
        pass

    @abstractmethod
    def transforms(self, x):
        pass


class DataParallelWithDict(nn.DataParallel):
    def gather(self, outputs, output_device):
        return {
            key: nn.parallel.gather([d[key] for d in outputs], output_device)
            for key in outputs[0]
        }


class GATEImageTextEncoder(GATEncoder):
    def __init__(
        self,
        image_embedding: GATEImageEncoder,
        text_embedding: GATETextEncoder,
        image_size: Optional[int] = None,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.image_size = image_size
        self.num_projection_features = num_projection_features

        if torch.cuda.device_count() > 1:

            transform_copy = deepcopy(self.image_embedding.transforms)
            num_features = self.image_embedding.num_features
            num_raw_features = self.image_embedding.num_raw_features
            self.image_embedding = self.image_embedding.to(
                torch.cuda.current_device()
            )
            if hasattr(self.image_embedding, "projection_layer"):
                projection_layer = self.image_embedding.projection_layer

            self.image_embedding = DataParallelWithDict(self.image_embedding)
            setattr(
                self.image_embedding,
                "transforms",
                transform_copy,
            )
            setattr(self.image_embedding, "num_features", num_features)
            setattr(self.image_embedding, "num_raw_features", num_raw_features)
            setattr(self.image_embedding, "projection_layer", projection_layer)

    @property
    def image_shape(self):
        return (self.image_size, self.image_size)

    @property
    def num_in_features_image(self):
        return self.image_embedding.num_features

    @property
    def num_in_features_text(self):
        return self.text_embedding.num_features

    @property
    def num_raw_features_image(self):
        return self.image_embedding.num_raw_features

    @property
    def num_raw_features_text(self):
        return self.text_embedding.num_raw_features

    @property
    def num_in_features_video(self):
        raise NotImplementedError(f"TimmCLIP does not have a video backbone")

    def init_weights(self):
        simple_init(self)

    def process_images(self, image: torch.Tensor) -> torch.Tensor:
        if image is None:
            raise ValueError("Image cannot be None.")

        # print(f"image shape: {image.shape}, device type: {image.device}")

        # for name, param in self.image_embedding.named_parameters():
        #     print(
        #         f"name: {name}, param shape: {param.shape}, device type: {param.device}"
        #     )

        return self.image_embedding(image)

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
            output_dict["image"] = self.process_images(image)
            if self.num_projection_features:
                output_dict["image"]["features"] = (
                    self.image_embedding.projection_layer(
                        output_dict["image"]["features"]
                    )
                )

        if video is not None:
            if len(video.shape) == 5:
                b, s, c, h, w = video.shape

                output_dict["video"] = self.process_images(
                    video.view(b * s, c, h, w)
                )
                if self.num_projection_features:
                    output_dict["video"]["features"] = (
                        self.image_embedding.projection_layer(
                            output_dict["video"]["features"]
                        )
                    )

                for k, v in output_dict["video"].items():
                    if v is not None:
                        if isinstance(v, list) or isinstance(v, tuple):
                            v = torch.stack(v, dim=2)
                        output_dict["video"][k] = v.view(b, s, *v.shape[1:])
            else:
                output_dict["video"] = self.process_images(video)
                if self.num_projection_features:
                    output_dict["video"]["features"] = (
                        self.image_embedding.projection_layer(
                            output_dict["video"]["features"]
                        )
                    )

        if text is not None:
            text: CLIPOutput = self.text_embedding(input_ids=text)
            output_dict["text"]["features"] = text.pooler_output
            output_dict["text"]["raw_features"] = text.last_hidden_state
            if self.num_projection_features:
                output_dict["text"]["features"] = (
                    self.text_embedding.projection_layer(
                        output_dict["text"]["features"]
                    )
                )

        return output_dict

    def get_transforms(self):
        def image_transforms(x):
            x = self.image_embedding.transforms(x)
            return x

        def text_transforms(x):
            return self.text_embedding.transforms(x)

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

    def get_image_encoder(self):
        return self.image_embedding

    def get_text_encoder(self):
        return self.text_embedding
