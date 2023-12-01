from gate.menu.core import (
    AdapterTypeNames,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)

dataset_configs = {
    dataset_name: dataset_name.value
    for dataset_name in DatasetName.MEDICAL_CLASSIFICATION.value
}

BATCH_SIZE = 32
ENCODER_CONFIG = AdapterTypeNames.IMAGE_CLASSIFICATION.value
RESNET_LR = 1e-3
VIT_LR = 1e-5
TRAINER_NAME = TrainerName.MULTI_CLASS_CLASSIFICATION.value
EVALUATOR_NAME = EvaluatorName.MULTI_CLASS_CLASSIFICATION.value


config = {
    "dataset": dataset_configs,
    "model": get_model_selection(
        adapter_config=ENCODER_CONFIG,
        batch_size=BATCH_SIZE,
        resnet_lr=RESNET_LR,
        vit_lr=VIT_LR,
    ),
    "trainer": TRAINER_NAME,
    "evaluator": EVALUATOR_NAME,
}
