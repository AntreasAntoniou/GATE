from gate.tiers.core import (
    Adapters,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)


class Config:
    BATCH_SIZE = 32
    ENCODER_CONFIG = Adapters.MULTI_CLASS_CLASSIFICATION.value
    RESNET_LR = 1e-3
    VIT_LR = 1e-5
    TRAINER_NAME = TrainerName.MULTI_CLASS_CLASSIFICATION.value
    EVALUATOR_NAME = EvaluatorName.MULTI_CLASS_CLASSIFICATION.value
    IMAGE_SIZE = 224

    def __init__(self):
        self.dataset = {
            dataset_name: dataset_name.value
            for dataset_name in DatasetName.MEDICAL_CLASSIFICATION.value
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
