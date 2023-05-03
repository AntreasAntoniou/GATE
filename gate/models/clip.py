from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from gate.boilerplate.decorators import configurable
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.classification import CLIPWithLinear


@dataclass
class ModelAndTransform:
    model: nn.Module
    transform: Any


@configurable
def build_model(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    num_classes: int = 100,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    from transformers import CLIPModel, CLIPProcessor

    feature_extractor: CLIPProcessor = CLIPProcessor.from_pretrained(
        model_name
    )
    feature_model = CLIPModel.from_pretrained(
        model_name, num_labels=num_classes
    )
    model = CLIPWithLinear(
        feature_model.vision_model, feature_model.vision_embed_dim, num_classes
    )

    if not pretrained:
        model.init_weights()

    model_modality_config_image_classification = TargetModalityConfig(
        image=[SourceModalityConfig(image=True)]
    )

    model_key_remapper_dict_config = {"image": "pixel_values"}

    gate_model = GATEModel.__config__(
        config=model_modality_config_image_classification,
        model=model,
        key_remapper_dict=model_key_remapper_dict_config,
    )

    transform = lambda image: feature_extractor(
        images=image, return_tensors="pt"
    )

    def transform_wrapper(input_dict: Dict):
        return {
            "pixel_values": transform(input_dict["image"])["pixel_values"][0],
            "labels": input_dict["labels"],
        }

    return ModelAndTransform(model=gate_model, transform=transform_wrapper)


if __name__ == "__main__":
    import accelerate
    import torch.nn.functional as F
    from rich import print

    model_and_transform = build_model()

    x_dummy = torch.rand(16, 3, 224, 224)
    y_dummy = torch.randint(0, 100, (16,))

    model = model_and_transform.model
    transform: Any = model_and_transform.transform

    accelerator = accelerate.Accelerator()

    x_dummy = x_dummy.to(accelerator.device)
    y_dummy = y_dummy.to(accelerator.device)
    model = accelerator.prepare(model)
    transform = accelerator.prepare(transform)

    fprop = model.forward(
        {
            "pixel_values": torch.cat(
                [
                    transform({"image": x, "labels": y})["input_images"].to(
                        accelerator.device
                    )
                    for x, y in zip(x_dummy, y_dummy)
                ],
                dim=0,
            ),
        }
    )

    print(f"Model output shape: {fprop.shape}")

    loss = F.cross_entropy(fprop, y_dummy)
    accelerator.backward(loss)
    print(f"Loss: {loss.item():.4f}")
