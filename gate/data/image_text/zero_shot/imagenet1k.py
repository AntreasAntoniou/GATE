# imagenet1k.py
from dataclasses import dataclass
from typing import Any, Optional


from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import build_dataset
from gate.data.image_text.zero_shot.metadata.clip_imagenet_prompts import (
    imagenet_prompt_templates,
)
from gate.data.image.classification.metadata.clip_imagenet_class_labels import (
    imagenet_classes,
)
from gate.data.tasks.zero_shot_classification import (
    ZeroShotViaLabelDescriptionTask,
)


@configurable(
    group="dataset",
    name="imagenet1k-zero-shot",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_imagenet1k_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=1000,
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=ZeroShotViaLabelDescriptionTask(
            prompt_templates=imagenet_prompt_templates,
            label_map=imagenet_classes,
        ),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=ZeroShotViaLabelDescriptionTask(
            prompt_templates=imagenet_prompt_templates,
            label_map=imagenet_classes,
        ),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=ZeroShotViaLabelDescriptionTask(
            prompt_templates=imagenet_prompt_templates,
            label_map=imagenet_classes,
        ),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_imagenet1k_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
