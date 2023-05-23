import pathlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import learn2learn as l2l
import PIL
import torch
from torchvision import transforms as T

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.few_shot import bytes_to_string
from gate.data.few_shot.core import FewShotClassificationMetaDataset
from gate.data.few_shot.utils import FewShotSuperSplitSetOptions
from gate.data.transforms.tiny_image_transforms import pad_image

logger = get_logger(
    __name__,
)


class FC100FewShotClassificationDataset(FewShotClassificationMetaDataset):
    def __init__(
        self,
        dataset_root: Union[str, pathlib.Path],
        split_name: str,
        download: bool,
        num_episodes: int,
        min_num_classes_per_set: int,
        min_num_samples_per_class: int,
        min_num_queries_per_class: int,
        num_classes_per_set: int,  # n_way
        num_samples_per_class: int,  # n_shot
        num_queries_per_class: int,
        variable_num_samples_per_class: bool,
        variable_num_classes_per_set: bool,
        support_set_input_transform: Optional[Any],
        query_set_input_transform: Optional[Any],
        support_set_target_transform: Optional[Any] = None,
        query_set_target_transform: Optional[Any] = None,
    ):
        DATASET_NAME = "metadataset/fc100"
        super(FC100FewShotClassificationDataset, self).__init__(
            dataset_name=DATASET_NAME,
            dataset_root=dataset_root,
            dataset_class=lambda set_name: l2l.vision.datasets.FC100(
                root=dataset_root,
                mode=set_name,
                download=download,
            ),
            preprocess_transforms=preprocess_transforms,
            split_name=split_name,
            num_episodes=num_episodes,
            num_classes_per_set=num_classes_per_set,
            num_samples_per_class=num_samples_per_class,
            num_queries_per_class=num_queries_per_class,
            variable_num_samples_per_class=variable_num_samples_per_class,
            variable_num_classes_per_set=variable_num_classes_per_set,
            input_target_annotation_keys=dict(
                inputs="image",
                targets="label",
                target_annotations="label",
            ),
            support_set_input_transform=support_set_input_transform,
            query_set_input_transform=query_set_input_transform,
            support_set_target_transform=support_set_target_transform,
            query_set_target_transform=query_set_target_transform,
            split_percentage={
                FewShotSuperSplitSetOptions.TRAIN: 64,
                FewShotSuperSplitSetOptions.VAL: 16,
                FewShotSuperSplitSetOptions.TEST: 20,
            },
            # split_config=l2l.vision.datasets.fgvc_fungi.SPLITS,
            subset_split_name_list=["train", "validation", "test"],
            label_extractor_fn=lambda x: bytes_to_string(x),
            min_num_classes_per_set=min_num_classes_per_set,
            min_num_samples_per_class=min_num_samples_per_class,
            min_num_queries_per_class=min_num_queries_per_class,
        )


def convert_single_to_three_channel_maybe(image):
    if not isinstance(image, torch.Tensor):
        image = T.ToTensor()(image)
    single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))
    if image.shape[0] == 1:
        image = single_to_three_channel(image)
    if not isinstance(image, PIL.Image.Image):
        image = T.ToPILImage()(image)
    return image


def preprocess_transforms(sample: Tuple):
    image = convert_single_to_three_channel_maybe(
        T.Resize(size=(224, 224))(sample[0])
    )
    label = sample[1]
    return {"image": image, "label": label}


def build_dataset(set_name: str, num_episodes: int, data_dir: str) -> dict:
    """
    Build a SVHN dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """

    if set_name not in ["train", "val", "test"]:
        raise KeyError(f"Invalid set name: {set_name}")

    dataset_root = pathlib.Path(data_dir)
    data_set = FC100FewShotClassificationDataset(
        dataset_root=dataset_root,
        split_name=set_name,
        download=True,
        num_episodes=num_episodes,
        min_num_classes_per_set=5,
        min_num_samples_per_class=2,
        min_num_queries_per_class=2,
        num_classes_per_set=50,
        num_samples_per_class=15,
        num_queries_per_class=15,
        variable_num_samples_per_class=True,
        variable_num_classes_per_set=True,
        support_set_input_transform=None,
        query_set_input_transform=None,
        support_set_target_transform=None,
        query_set_target_transform=None,
    )

    return data_set


from rich import print


# def key_mapper(input_dict):
#     return {
#         "image": input_dict["image"],
#         "labels": input_dict["labels"],
#     }


def key_mapper(input_dict):
    input_dict["image"]["image"]["support_set"] = [
        T.ToPILImage()(item)
        for item in input_dict["image"]["image"]["support_set"]
    ]

    input_dict["image"]["image"]["query_set"] = [
        T.ToPILImage()(item)
        for item in input_dict["image"]["image"]["query_set"]
    ]

    return {
        "image": input_dict["image"],
        "labels": input_dict["labels"],
    }


@configurable(
    group="dataset",
    name="fc100-fs-classification",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir, num_episodes=10000),
        infinite_sampling=True,
        item_keys=["image", "labels"],
        transforms=[key_mapper, transforms],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir, num_episodes=600),
        infinite_sampling=False,
        item_keys=["image", "labels"],
        transforms=[key_mapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir, num_episodes=600),
        infinite_sampling=False,
        item_keys=["image", "labels"],
        transforms=[key_mapper, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
