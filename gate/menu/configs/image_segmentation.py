from rich import print

from gate.menu.core import (
    AdapterConfig,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)
from gate.models.task_adapters.semantic_segmentation import (
    SegmentationLossOptions,
)


class Config:
    BATCH_SIZE = 8
    ENCODER_CONFIG = [
        AdapterConfig(
            adapter_name="segmentation-adapter",
            loss_type_id=SegmentationLossOptions.DEFAULT.value,
            background_loss_weight=0.1,
            dice_loss_weight=dice_loss_weight,
            focal_loss_weight=focal_loss_weight,
            ce_loss_weight=ce_loss_weight,
        )
        for (dice_loss_weight, focal_loss_weight, ce_loss_weight) in [
            # (1.0, 1.0, 1.0),
            # (0.0, 1.0, 1.0),
            # (1.0, 0.0, 1.0),
            # (1.0, 1.0, 0.0),
            # (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            # (1.0, 0.0, 0.0),
        ]
    ]  # 7 different adapter configurations
    RESNET_LR = 6e-4
    VIT_LR = 1e-5
    TRAINER_NAME = TrainerName.IMAGE_SEMANTIC_SEGMENTATION.value
    EVALUATOR_NAME = EvaluatorName.IMAGE_SEMANTIC_SEGMENTATION.value
    IMAGE_SIZE = 1024
    WEIGHT_DECAY = 0.01

    def __init__(self):
        self.dataset = {
            dataset_name: dataset_name.value
            for dataset_name in DatasetName.IMAGE_SEGMENTATION.value
        }
        self.model = [
            get_model_selection(
                adapter_config=adapter_config,
                batch_size=self.BATCH_SIZE,
                resnet_lr=self.RESNET_LR,
                vit_lr=self.VIT_LR,
                image_size=self.IMAGE_SIZE,
                wd=self.WEIGHT_DECAY,
            )
            for adapter_config in self.ENCODER_CONFIG
        ]
        self.model = {
            f"{adapter_name}-hpo-{idx}": adapter_config
            for idx, model in enumerate(self.model)
            for adapter_name, adapter_config in model.items()
        }
        self.trainer = self.TRAINER_NAME
        self.evaluator = self.EVALUATOR_NAME
        self.image_size = self.IMAGE_SIZE
        self.wd = self.WEIGHT_DECAY

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
