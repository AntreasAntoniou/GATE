from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.models.task_adapters import BaseModule
from gate.models.task_adapters.utils import get_similarities
from tqdm.auto import tqdm


class DuoModalZeroShotModel(BaseModule):
    def __init__(
        self,
        modality_a_model: nn.Module,
        modality_b_model: nn.Module,
        modality_a_identifier: str,
        modality_b_identifier: str,
        modality_a_num_features: int,
        modality_b_num_features: int,
        projection_num_features: Optional[int] = None,
        temperature_parameter: Optional[float] = 1.0,
        head_identifier: Optional[str] = "projection_output",
    ):
        super().__init__()
        self.modality_a_model = modality_a_model
        self.modality_b_model = modality_b_model

        self.modality_a_identifier = modality_a_identifier
        self.modality_b_identifier = modality_b_identifier

        self.head_identifier = head_identifier

        self.projection_num_features = projection_num_features
        if temperature_parameter is None:
            self.temperature_parameter = nn.Parameter(torch.tensor(1.0))
        else:
            self.temperature_parameter = temperature_parameter

        if self.projection_num_features is not None:
            self.modality_a_linear = nn.Linear(
                modality_a_num_features, projection_num_features, bias=False
            )
            self.modality_b_linear = nn.Linear(
                modality_b_num_features, projection_num_features, bias=False
            )

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # check that only two modalities are passed
        modalities = [image, text, audio, video]
        non_none_modalities = [
            modality for modality in modalities if modality is not None
        ]

        if len(non_none_modalities) != 2:
            raise ValueError(
                f"Exactly two modalities must be provided. You provided {len(non_none_modalities)}"
            )

        modality_a_features = None
        modality_b_features = None

        if image is not None:
            if self.modality_a_identifier == "image":
                modality_a_features = self.modality_a_model(image=image)[
                    self.modality_a_identifier
                ][self.head_identifier]
            elif self.modality_b_identifier == "image":
                modality_b_features = self.modality_b_model(image=image)[
                    self.modality_b_identifier
                ][self.head_identifier]

        if text is not None:
            if self.modality_a_identifier == "text":
                modality_a_features = self.modality_a_model(text=text)[
                    self.modality_a_identifier
                ][self.head_identifier]
            elif self.modality_b_identifier == "text":
                modality_b_features = self.modality_b_model(text=text)[
                    self.modality_b_identifier
                ][self.head_identifier]
        if audio is not None:
            if self.modality_a_identifier == "audio":
                modality_a_features = self.modality_a_model(audio=audio)[
                    self.modality_a_identifier
                ][self.head_identifier]
            elif self.modality_b_identifier == "audio":
                modality_b_features = self.modality_b_model(audio=audio)[
                    self.modality_b_identifier
                ][self.head_identifier]

        if video is not None:
            if self.modality_a_identifier == "video":
                modality_a_features = self.modality_a_model(video=video)[
                    self.modality_a_identifier
                ][self.head_identifier]
            elif self.modality_b_identifier == "video":
                modality_b_features = self.modality_b_model(video=video)[
                    self.modality_b_identifier
                ][self.head_identifier]

        if self.projection_num_features is not None:
            modality_a_features = self.modality_a_linear(modality_a_features)
            modality_b_features = self.modality_b_linear(modality_b_features)

        metrics_dict = get_similarities(
            modality_a_name=self.modality_a_identifier,
            modality_a_features=modality_a_features,
            modality_b_name=self.modality_b_identifier,
            modality_b_features=modality_b_features,
            temperature_parameter=self.temperature_parameter,
            return_loss=return_loss,
        )

        losses_list = [
            value for key, value in metrics_dict.items() if "loss" in key
        ]
        if len(losses_list) > 0:
            loss = torch.mean(torch.stack(losses_list))
            metrics_dict["loss"] = loss
        return metrics_dict


from accelerate import Accelerator

accelerator = Accelerator()


class DuoModalZeroShotModelWithPresetClasses(BaseModule):
    def __init__(
        self,
        image_modality_model: nn.Module,
        text_modality_model: nn.Module,
        image_modality_num_features: int,
        class_prompts: Dict[str, List[str]] = None,
        projection_num_features: Optional[int] = None,
        temperature_parameter: Optional[float] = 1.0,
        backbone_output_key: str = "projection_output",
    ):
        super().__init__()
        self.image_modality_model = image_modality_model
        self.text_modality_model = text_modality_model
        self.backbone_output_key = backbone_output_key

        self.projection_num_features = projection_num_features

        if temperature_parameter is None:
            self.temperature_parameter = nn.Parameter(torch.tensor(1.0))
        else:
            self.temperature_parameter = temperature_parameter

        self.class_prompts = class_prompts

        if self.projection_num_features is not None:
            self.linear_projection = nn.Linear(
                image_modality_num_features,
                projection_num_features,
                bias=False,
            )
        self.class_prototypes = None

    def build_class_prototypes(self, class_prompts):
        self.class_prototypes = []
        self.text_modality_model.eval()
        print(f"Building class prototypes for {len(class_prompts)} classes")
        with torch.no_grad():
            for class_key, class_prompts in tqdm(class_prompts.items()):
                class_prompt_tokens = (
                    self.text_modality_model.get_transforms()["text"](
                        class_prompts
                    )
                )
                class_prompt_tokens = class_prompt_tokens.to(
                    accelerator.device
                )

                class_prototype = self.text_modality_model(
                    text=class_prompt_tokens
                )["text"][self.backbone_output_key].mean(0)
                self.class_prototypes.append(class_prototype)

        self.class_prototypes = torch.stack(self.class_prototypes)

    @torch.inference_mode()
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.class_prototypes is None:
            self.build_class_prototypes(self.class_prompts)

        if image is not None:
            image_features = self.image_modality_model(image=image)["image"][
                self.backbone_output_key
            ]
        else:
            raise ValueError("An image input must be provided")

        if self.projection_num_features is not None:
            image_features = self.modality_a_linear(image_features)

        logits = (
            F.linear(image_features, self.class_prototypes)
            * self.temperature_parameter
        )

        output_dict: dict[str, torch.Tensor] = {"logits": logits}

        return output_dict
