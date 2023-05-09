from collections import defaultdict
import pathlib
from typing import Any, Dict, Optional, Union
from tali.utils import download_model_with_name

import torch
import torch.nn as nn

from tali.models import TALIModel, MultiModalityConfig
from transformers import CLIPProcessor, WhisperProcessor

import accelerate
from rich import print

from gate.boilerplate.utils import (
    create_hf_model_repo_and_download_maybe,
    download_model_checkpoint_from_hub,
)


class TALINet(nn.Module):
    """
    TALINet is a multi-modal model that can process image, text, audio and video data.
    It uses separate models for each modality, and merges their outputs.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        whisper_model_name: str = "openai/whisper-small",
        pretrained: bool = True,
    ):
        super().__init__()

        # Initialize TALIModel with specified image, text and audio models
        self.model = TALIModel(
            image_text_model_name=clip_model_name,
            audio_model_name=whisper_model_name,
            multi_modality_config=MultiModalityConfig(),
        )

        self.image_text_preprocessor: CLIPProcessor = (
            CLIPProcessor.from_pretrained(clip_model_name)
        )

        self.audio_preprocessor = WhisperProcessor.from_pretrained(
            whisper_model_name
        )

        self.video_num_features = self.model.video_linear_layer.in_features
        self.image_num_features = self.model.image_linear_layer.in_features
        self.text_num_features = self.model.text_linear_layer.in_features
        self.audio_num_features = self.model.audio_linear_layer.in_features

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model. Processes each modality if provided, and merges the outputs.

        Args:
            image (Optional[torch.Tensor]): The image tensor.
            text (Optional[torch.Tensor]): The text tensor.
            audio (Optional[torch.Tensor]): The audio tensor.
            video (Optional[torch.Tensor]): The video tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the output tensors from each modality.
        """

        # Raise ValueError if no input modality is provided
        if image is None and text is None and audio is None and video is None:
            raise ValueError(
                f"🚫 Must provide at least one input modality to {self.__class__.__name__}."
            )

        # Process each modality and merge the outputs
        output_dict = defaultdict(dict)

        # For each modality, call the corresponding forward method and merge the results into output_dict
        # 💡 Using dictionary comprehension to simplify code and improve readability
        if image is not None:
            output_dict |= {
                f"image_{k}": v
                for k, v in self.model.forward_image(image).items()
            }
        if text is not None:
            output_dict |= {
                f"text_{k}": v
                for k, v in self.model.forward_text(text).items()
            }
        if audio is not None:
            output_dict |= {
                f"audio_{k}": v
                for k, v in self.model.forward_audio(audio).items()
            }
        if video is not None:
            output_dict |= {
                f"video_{k}": v
                for k, v in self.model.forward_video(video).items()
            }

        return output_dict

    def get_transforms(self):
        return {
            "image": lambda x: self.image_text_preprocessor(
                images=x, return_tensors="pt"
            ).pixel_values.squeeze(1),
            "text": lambda x: self.image_text_preprocessor(
                text=x, return_tensors="pt", padding=True, truncation=True
            ).input_ids,
            "audio": lambda x: torch.cat(
                [
                    self.audio_preprocessor(
                        item.view(-1),
                        sampling_rate=16000,
                        return_tensors="pt",
                    ).input_features
                    for item in x.unbind(0)
                ]
            ),
            "video": lambda x: torch.stack(
                [
                    self.image_text_preprocessor(
                        images=image, return_tensors="pt"
                    ).pixel_values
                    for image in x
                ],
                dim=0,
            ),
        }

    def load_from_hub(
        self, model_repo_path: str, checkpoint_identifier: str, **kwargs
    ):
        import os

        download_dir, repo_path = download_model_checkpoint_from_hub(
            hf_repo_path=model_repo_path,
            hf_cache_dir=os.environ["HF_CACHE_DIR"],
            checkpoint_identifier=checkpoint_identifier
            if "latest" != checkpoint_identifier
            else None,
            get_latest=True if "latest" == checkpoint_identifier else False,
        )
        self.accelerator = accelerate.Accelerator()
        self.model = self.accelerator.prepare(self.model)
        # Load the state dict from the path
        state_dict = torch.load(
            pathlib.Path(download_dir) / "pytorch_model.bin"
        )
        # Load the state dict into the model
        report = self.load_state_dict(state_dict, strict=False)
        print(report)


if __name__ == "__main__":
    model = TALINet()
    model.load_from_hub(
        model_repo_path="Antreas/tali-2-tali_image_text_base_patch16_224-wit_image_text_dataset-2306",
        checkpoint_identifier="latest",
    )
    print(model)
