from gate.menu.core import (
    AdapterTypeNames,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)

dataset_configs = DatasetName.IMAGE_TEXT_ZERO_SHOT_CLASSIFICATION.value


dataset_configs = {
    dataset_name: dataset_name.value for dataset_name in dataset_configs
}

BATCH_SIZE = 64
ENCODER_CONFIG = AdapterTypeNames.ZERO_SHOT_IMAGE_TEXT.value
RESNET_LR = 1e-3
VIT_LR = 1e-5
TRAINER_NAME = TrainerName.IMAGE_TO_TEXT_ZERO_SHOT_CLASSIFICATION.value
EVALUATOR_NAME = EvaluatorName.IMAGE_TO_TEXT_ZERO_SHOT_CLASSIFICATION.value
IMAGE_SIZE = 224

config = {
    "dataset": dataset_configs,
    "model": get_model_selection(
        adapter_config=ENCODER_CONFIG,
        batch_size=BATCH_SIZE,
        resnet_lr=RESNET_LR,
        vit_lr=VIT_LR,
        image_size=IMAGE_SIZE,
    ),
    "trainer": TRAINER_NAME,
    "evaluator": EVALUATOR_NAME,
}
