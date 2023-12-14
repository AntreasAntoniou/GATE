import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.models.backbones import GATEncoder
from gate.models.core import SourceModalityConfig, TargetModalityConfig
from gate.models.task_adapters import BaseModule
from gate.models.task_adapters.utils import (
    compute_zero_shot_loss_and_metrics,
    get_similarities,
)

accelerator = Accelerator()


logger = logging.getLogger(__name__)


@configurable(
    group="adapter",
    name="duo-modal-zero-shot-classifier",
)
class DuoModalZeroShotModel(BaseModule):
    def __init__(
        self,
        encoder: GATEncoder,
        projection_num_features: Optional[int] = 768,
        temperature_parameter: Optional[float] = 1.0 / 0.07,
        head_identifier: Optional[str] = "features",
    ):
        super().__init__()
        self.encoder = encoder

        self.head_identifier = head_identifier

        self.projection_num_features = projection_num_features
        if temperature_parameter is None:
            self.temperature_parameter = nn.Parameter(torch.tensor(1.0 / 0.07))
        else:
            self.temperature_parameter = nn.Parameter(
                torch.tensor(temperature_parameter)
            )

        if self.projection_num_features is not None:
            self.image_linear_projection = nn.Linear(
                in_features=encoder.num_in_features_image,
                out_features=projection_num_features,
                bias=False,
            )
            self.text_linear_projection = nn.Linear(
                in_features=encoder.num_in_features_text,
                out_features=projection_num_features,
                bias=False,
            )
        self.build()

    def build(self):
        dummy_batch = {
            "image": torch.randn(
                2, 3, self.encoder.image_shape[0], self.encoder.image_shape[1]
            ),
            "text": torch.randint(0, 100, (2, 10)),
        }
        _ = self(**dummy_batch)

    @ensemble_marker
    def compute_loss_and_metrics(self, logits, **kwargs):
        return compute_zero_shot_loss_and_metrics(
            similarities=logits["similarities"],
            is_irregular_shape=logits["is_irregular_shape"],
        )

    @property
    def encoder_transforms(self):
        return self.encoder.get_transforms()

    @property
    def modality_config(self):
        return TargetModalityConfig(
            image_text=[SourceModalityConfig(image=True, text=True)]
        )

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        return_loss: bool = True,
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

        image_features = None
        text_features = None

        is_irregular_shape = False

        if len(image.shape) == 5:
            image = image.view(-1, *image.shape[2:])
            is_irregular_shape = True

        if len(text.shape) == 3:
            text = text.view(-1, *text.shape[2:])
            is_irregular_shape = True

        if image is not None:
            image_features = self.encoder(image=image)["image"][
                self.head_identifier
            ]

        if text is not None:
            text_features = self.encoder(text=text)["text"][
                self.head_identifier
            ]

        if self.projection_num_features is not None:
            image_features = self.image_linear_projection(image_features)
            text_features = self.text_linear_projection(text_features)

        similarities_dict = get_similarities(
            image_features=image_features,
            text_features=text_features,
            temperature_parameter=self.temperature_parameter,
        )

        output_dict = {
            "logits": {
                "similarities": similarities_dict,
                "is_irregular_shape": is_irregular_shape,
            },
            "labels": similarities_dict,
        }

        metrics_dict = self.compute_loss_and_metrics(
            logits=output_dict["logits"]
        )

        losses_list = [
            value for key, value in metrics_dict.items() if "loss" in key
        ]
        if len(losses_list) > 0 and return_loss:
            loss = torch.mean(torch.stack(losses_list))
            metrics_dict["loss"] = loss

        return output_dict | metrics_dict

    def adapter_transforms(self, inputs: Union[Dict, Any]):
        if "image" in inputs:
            image = inputs["image"]
            if isinstance(image, List):
                image = torch.stack(
                    [
                        self.encoder_transforms["image"](sample)
                        for sample in image
                    ]
                )
            else:
                image = self.encoder_transforms["image"](image)
            inputs["image"] = image

        if "text" in inputs:
            text = inputs["text"]
            text = self.encoder_transforms["text"](text)

            inputs["text"] = text

        return inputs
