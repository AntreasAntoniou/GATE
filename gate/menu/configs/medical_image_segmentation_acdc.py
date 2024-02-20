from gate.menu.core import (
    Adapters,
    DatasetName,
    EvaluatorName,
    MixedPrecisionMode,
    TrainerName,
    get_model_selection,
)


class Config:
    BATCH_SIZE = 1
    ENCODER_CONFIG = Adapters.MD_SEGMENTATION.value
    RESNET_LR = 6e-4
    VIT_LR = 1e-5
    TRAINER_NAME = TrainerName.MEDICAL_SEMANTIC_SEGMENTATION.value
    EVALUATOR_NAME = EvaluatorName.MEDICAL_SEMANTIC_SEGMENTATION.value
    IMAGE_SIZE = 512
    WEIGHT_DECAY = 0.01

    def __init__(self):
        self.dataset = {
            dataset_name: dataset_name.value
            for dataset_name in DatasetName.MEDICAL_ACDC_SEGMENTATION.value
        }

        self.model = get_model_selection(
            adapter_config=self.ENCODER_CONFIG,
            batch_size=self.BATCH_SIZE,
            resnet_lr=self.RESNET_LR,
            vit_lr=self.VIT_LR,
            image_size=self.IMAGE_SIZE,
            wd=self.WEIGHT_DECAY,
            mixed_precision_mode=MixedPrecisionMode.FP16,
        )
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
