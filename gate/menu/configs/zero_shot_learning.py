from gate.menu.core import (
    AdapterTypeNames,
    DatasetName,
    EncoderNames,
    EvaluatorName,
    LearningRateConfig,
    ModelConfig,
    TrainerName,
    get_model_selection,
)

dataset_configs = DatasetName.IMAGE_TEXT_ZERO_SHOT_CLASSIFICATION.value


dataset_configs = {
    dataset_name: dataset_name.value for dataset_name in dataset_configs
}

BATCH_SIZE = 64
MODEL_TYPE = AdapterTypeNames.TIMM_ZERO_SHOT_IMAGE_TEXT.value
RESNET_LR = 1e-3
VIT_LR = 1e-5
TRAINER_NAME = TrainerName.IMAGE_TO_TEXT_ZERO_SHOT_CLASSIFICATION.value
EVALUATOR_NAME = EvaluatorName.IMAGE_TO_TEXT_ZERO_SHOT_CLASSIFICATION.value

config = {
    "dataset": dataset_configs,
    "model": get_model_selection(
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        resnet_lr=RESNET_LR,
        vit_lr=VIT_LR,
    ),
    "trainer": TRAINER_NAME,
    "evaluator": EVALUATOR_NAME,
}
