import logging
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import transforms as T
from transformers import CLIPModel, CLIPProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import (
    CLIPEncoder,
    CLIPTextEmbeddings,
    CLIPVisionEmbeddings,
    _expand_mask,
    _make_causal_mask,
)

from gate.models.backbones import Modality, apply_preprocessing_transforms
from gate.models.core import reinit
from gate.models.task_adapters.modality_transfer_classification import (
    VisionRootReplacedBackbone,
)

logger = logging.getLogger(__name__)


def forward_dict(self, x):
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


class ModifiedCLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(
            embed_dim, eps=config.layer_norm_eps
        )

    def forward(
        self,
        image: Optional[torch.Tensor],
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = self.config.output_attentions

        output_hidden_states = self.config.output_hidden_states

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        hidden_states = self.embeddings(input_ids=image, position_ids=None)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(
            hidden_states.shape[:2],
            hidden_states.dtype,
            device=hidden_states.device,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[-1]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "features": pooled_output,
            "raw_features": last_hidden_state,
            "per_layer_raw_features": hidden_states,
        }


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


class CLIPTextAdapter(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.clip = CLIPModel.from_pretrained(model_name)
        self.text_transforms = TextProcessor(self.preprocessor)

        if not pretrained:
            self.clip.init_weights()

        vision_embedding = ModifiedCLIPTextTransformer(
            self.clip.config.text_config
        )

        self.vision_model = VisionRootReplacedBackbone(
            model=vision_embedding,
            num_root_features=self.clip.text_embed_dim,
            backbone_root_layers_to_remove=["embeddings"],
            image_size=image_size,
            num_channels=3,
            patch_size=16,
            source_modality=Modality.image,
            target_modality=Modality.image,
        )
        self.visual_projection = self.clip.text_projection
        self.text_model = self.clip.text_model
        self.text_projection = self.clip.text_projection

        # setattr signature: setattr(object, name, value)

        setattr(self.vision_model, "legacy_forward", self.text_model.forward)
        setattr(self.text_model, "legacy_forward", self.text_model.forward)

        setattr(
            self.text_model, "forward", forward_dict.__get__(self.text_model)
        )

        self.image_num_features = self.clip.vision_embed_dim
        self.text_num_features = self.clip.text_embed_dim

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

        # self.model.forward expects
        # input_ids: Optional[torch.LongTensor] = None,
        # pixel_values: Optional[torch.FloatTensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # return_loss: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,

        if image is not None:
            output_dict["image"] = self.vision_model(image=image)

            output_dict["image"]["classifier"] = self.visual_projection(
                output_dict["image"]["features"]
            )

        if video is not None:
            if len(video.shape) == 5:
                b, s, c, h, w = video.shape
                output_dict["video"] = self.vision_model.forward(
                    video.view(b * s, c, h, w)
                )
                for k, v in output_dict["video"].items():
                    if v is not None:
                        if isinstance(v, list):
                            v = torch.stack(v, dim=2)

                        output_dict["video"][k] = v.view(b, s, *v.shape[1:])
            else:
                output_dict["video"] = self.vision_model.forward(video)

            output_dict["video"]["projection_output"] = self.visual_projection(
                output_dict["video"]["features"]
            )

        if text is not None:
            output_dict["text"] = self.text_model(x=text)
            output_dict["text"]["projection_output"] = self.text_projection(
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
