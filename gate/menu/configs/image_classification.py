from gate.menu.core import (
    AdapterTypeNames,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)

dataset_configs = {
    dataset_name: dataset_name.value
    for dataset_name in DatasetName.IMAGE_CLASSIFICATION.value
}

BATCH_SIZE = 64
ADAPTER_CONFIG = AdapterTypeNames.IMAGE_CLASSIFICATION.value
RESNET_LR = 1e-3
VIT_LR = 1e-5
TRAINER_NAME = TrainerName.IMAGE_CLASSIFICATION.value
EVALUATOR_NAME = EvaluatorName.IMAGE_CLASSIFICATION.value


config = {
    "dataset": dataset_configs,
    "model": get_model_selection(
        adapter_config=ADAPTER_CONFIG,
        batch_size=BATCH_SIZE,
        resnet_lr=RESNET_LR,
        vit_lr=VIT_LR,
    ),
    "trainer": TRAINER_NAME,
    "evaluator": EVALUATOR_NAME,
}
