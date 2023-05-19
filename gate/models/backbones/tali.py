import pathlib
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import accelerate
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig
from rich import print
from tali.models import MultiModalityConfig, TALIModel
from tali.utils import download_model_with_name
from transformers import CLIPProcessor, WhisperProcessor

from gate.boilerplate.utils import download_model_checkpoint_from_hub
from gate.models.backbones import (
    Modality,
    apply_preprocessing_transforms,
    image_dim_reshape,
)
from gate.models.core import reinit


class TALINet(nn.Module):
    """
    TALINet is a multi-modal model that can process image, text, audio and video data.
    It uses separate models for each modality, and merges their outputs.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        whisper_model_name: str = "openai/whisper-small",
        model_repo_path: Optional[
            str
        ] = "Antreas/tali-2-tali_image_text_base_patch16_224-wit_tali_image_text_dataset-2306",
        checkpoint_identifier: Optional[str] = "latest",
        pretrained: bool = True,
    ):
        super().__init__()

        # Initialize TALIModel with specified image, text and audio models
        # if pretrained:
        #     if checkpoint_identifier is not None:
        #         self.load_from_hub(
        #             model_repo_path=model_repo_path,
        #             checkpoint_identifier=checkpoint_identifier,
        #         )
        #     else:
        #         self.load_from_hub(
        #             model_repo_path=model_repo_path,
        #             checkpoint_identifier="latest",
        #         )
        # else:
        self.model = TALIModel(
            image_text_model_name=clip_model_name,
            audio_model_name=whisper_model_name,
            multi_modality_config=MultiModalityConfig(),
        )

        self.image_text_preprocessor: CLIPProcessor = (
            CLIPProcessor.from_pretrained(clip_model_name)
        )

        self.tokenizer = self.image_text_preprocessor

        self.audio_preprocessor = WhisperProcessor.from_pretrained(
            whisper_model_name
        )

        if hasattr(self.model, "video_linear_layer"):
            self.video_num_features = self.model.video_linear_layer.in_features

        if hasattr(self.model, "image_linear_layer"):
            self.image_num_features = self.model.image_linear_layer.in_features
            self.image_num_patches = 14 * 14

        if hasattr(self.model, "text_linear_layer"):
            self.text_num_features = self.model.text_linear_layer.in_features

        if hasattr(self.model, "audio_linear_layer"):
            self.audio_num_features = self.model.audio_linear_layer.in_features

        self.to("cpu")

    def init_weights(self):
        reinit(self)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model. Processes each modality if provided,
        and merges the outputs.

        Args:
            image (Optional[torch.Tensor]): The image tensor.
            text (Optional[torch.Tensor]): The text tensor.
            audio (Optional[torch.Tensor]): The audio tensor.
            video (Optional[torch.Tensor]): The video tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the output
            tensors from each modality.
        """

        # Raise ValueError if no input modality is provided
        if image is None and text is None and audio is None and video is None:
            raise ValueError(
                f"🚫 Must provide at least one "
                f"input modality to {self.__class__.__name__}."
            )

        # Process each modality and merge the outputs
        output_dict = defaultdict(dict)

        # For each modality, call the corresponding forward method
        # and merge the results into output_dict
        # 💡 Using dictionary comprehension to simplify code and
        # improve readability
        # if isinstance(image, Dict):
        #     image = image["image"]
        # if isinstance(text, Dict):
        #     text = text["text"]
        # if isinstance(audio, Dict):
        #     audio = audio["audio"]
        # if isinstance(video, Dict):
        #     video = video["video"]

        if image is not None:
            output_dict |= {
                "image": {
                    k: v for k, v in self.model.forward_image(image).items()
                },
            }
        if text is not None:
            output_dict |= {
                "text": {
                    k: v for k, v in self.model.forward_text(text).items()
                },
            }
        if audio is not None:
            output_dict |= {
                "audio": {
                    k: v for k, v in self.model.forward_audio(audio).items()
                },
            }
        if video is not None:
            output_dict |= {
                "video": {
                    k: v for k, v in self.model.forward_video(video).items()
                },
            }

        # print(f"🚀 Output dict: {output_dict}")

        return output_dict

    def get_transforms(self):
        def image_transforms(x):
            return self.image_text_preprocessor(
                images=x, return_tensors="pt"
            ).pixel_values

        def text_transforms(x):
            return self.image_text_preprocessor(
                text=x, return_tensors="pt", padding=True, truncation=True
            ).input_ids.squeeze(0)

        return {
            "image": lambda x: apply_preprocessing_transforms(
                x=x, transforms=image_transforms, modality=Modality.image
            ),
            "text": lambda x: apply_preprocessing_transforms(
                x=x, transforms=text_transforms, modality=Modality.text
            ),
        }

    # def get_transforms(self):
    #     def image_transforms(x):
    #         return self.image_text_preprocessor(
    #             images=x, return_tensors="pt"
    #         ).pixel_values

    #     def text_transforms(x):
    #         return self.image_text_preprocessor(
    #             text=x, return_tensors="pt", padding=True, truncation=True
    #         ).input_ids.squeeze(0)

    #     def audio_transforms(x):
    #         return torch.cat(
    #             [
    #                 self.audio_preprocessor(
    #                     item.view(-1),
    #                     sampling_rate=16000,
    #                     return_tensors="pt",
    #                 ).input_features
    #                 for item in x.unbind(0)
    #             ]
    #         )

    #     def video_transforms(x):
    #         return (
    #             torch.stack(
    #                 [
    #                     self.image_text_preprocessor(
    #                         images=image, return_tensors="pt"
    #                     ).pixel_values
    #                     for image in x
    #                 ],
    #                 dim=0,
    #             ),
    #         )

    #     return {
    #         "image": lambda x: apply_preprocessing_transforms(
    #             x=x, transforms=image_transforms, modality=Modality.image
    #         ),
    #         "text": lambda x: apply_preprocessing_transforms(
    #             x=x, transforms=text_transforms, modality=Modality.text
    #         ),
    #         "audio": lambda x: apply_preprocessing_transforms(
    #             x=x, transforms=audio_transforms, modality=Modality.audio
    #         ),
    #         "video": lambda x: apply_preprocessing_transforms(
    #             x=x, transforms=video_transforms, modality=Modality.video
    #         ),
    #     }

    def load_from_hub(
        self, model_repo_path: str, checkpoint_identifier: str, **kwargs
    ):
        import os

        download_dir = download_model_checkpoint_from_hub(
            hf_repo_path=model_repo_path,
            hf_cache_dir=os.environ["HF_CACHE_DIR"],
            checkpoint_identifier=checkpoint_identifier
            if "latest" != checkpoint_identifier
            else None,
            get_latest=True if "latest" == checkpoint_identifier else False,
        )
        if download_dir is not None:
            config = yaml.safe_load(open(download_dir["config_filepath"]))
            model_config = config["model"]
            del model_config["_target_"]
            model_config = DictConfig(model_config)
            self.model = self.model = TALIModel(**model_config)

            self.accelerator = accelerate.Accelerator()
            self.model = self.accelerator.prepare(self.model)
            # Load the state dict from the path
            state_dict = torch.load(download_dir["model_filepath"])
            # Load the state dict into the model
            report = self.model.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded weights succesfully, the reported outcome was: {report}"
            )


if __name__ == "__main__":
    model = TALINet(
        model_repo_path="Antreas/tali-2-tali_image_text_base_patch16_224-wit_image_text_dataset-2306",
        checkpoint_identifier="latest",
    )

    # print(model)
    # TODO:
    # 1. Get a way to build the right TALI model given config (10m) (DONE)
    # 2. Add a timm model with clip text encoder as another (15m) (DONE)
    # baseline so we can immediately get access to shitloads of baseline models
    # 3. Get VQA to work (2h)
    # 4. Get text benchmarks to work (2h)
    # 5. RUN ALL THE BENCHMARKS (2h)
