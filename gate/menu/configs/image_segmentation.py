from gate.menu.core import (
    Adapters,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)


class Config:
    BATCH_SIZE = 8
    ENCODER_CONFIG = Adapters.SEGMENTATION.value
    RESNET_LR = 6e-4
    VIT_LR = 6e-6
    TRAINER_NAME = TrainerName.IMAGE_SEMANTIC_SEGMENTATION.value
    EVALUATOR_NAME = EvaluatorName.IMAGE_SEMANTIC_SEGMENTATION.value
    IMAGE_SIZE = 1024

    def __init__(self):
        self.dataset = {
            dataset_name: dataset_name.value
            for dataset_name in DatasetName.IMAGE_SEGMENTATION.value
        }
        self.model = get_model_selection(
            adapter_config=self.ENCODER_CONFIG,
            batch_size=self.BATCH_SIZE,
            resnet_lr=self.RESNET_LR,
            vit_lr=self.VIT_LR,
            image_size=self.IMAGE_SIZE,
            wd=0.01,
        )
        self.trainer = self.TRAINER_NAME
        self.evaluator = self.EVALUATOR_NAME
        self.image_size = self.IMAGE_SIZE
        self.wd = 0.01

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
