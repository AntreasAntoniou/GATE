import pathlib
from typing import Any, Optional, Tuple, Union

import learn2learn as l2l
from torchvision import transforms

from gate.boilerplate.utils import get_logger
from gate.data.few_shot import bytes_to_string
from gate.data.few_shot.core import FewShotClassificationMetaDataset
from gate.data.few_shot.utils import FewShotSuperSplitSetOptions

logger = get_logger(
    __name__,
)


def preprocess_transforms(sample: Tuple):
    image_transforms = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    image = image_transforms(sample[0])
    label = sample[1]
    return {"image": image, "label": label}


class VGGFlowersFewShotClassificationDataset(FewShotClassificationMetaDataset):
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
        variable_num_queries_per_class: bool,
        variable_num_classes_per_set: bool,
        support_set_input_transform: Optional[Any],
        query_set_input_transform: Optional[Any],
        support_set_target_transform: Optional[Any] = None,
        query_set_target_transform: Optional[Any] = None,
    ):
        DATASET_NAME = "metadataset/vgg_flower_102"
        super(VGGFlowersFewShotClassificationDataset, self).__init__(
            dataset_name=DATASET_NAME,
            dataset_root=dataset_root,
            dataset_class=lambda set_name: l2l.vision.datasets.VGGFlower102(
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
            variable_num_queries_per_class=variable_num_queries_per_class,
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
