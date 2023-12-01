from gate.menu.core import (
    AdapterTypeNames,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)

dataset_configs = {
    dataset_name: dataset_name.value
    for dataset_name in DatasetName.IMAGE_SEGMENTATION.value
}

BATCH_SIZE = 8
MODEL_TYPE = AdapterTypeNames.TIMM_SEGMENTATION.value
RESNET_LR = 6e-4
VIT_LR = 6e-6
TRAINER_NAME = TrainerName.IMAGE_SEMANTIC_SEGMENTATION.value
EVALUATOR_NAME = EvaluatorName.IMAGE_SEMANTIC_SEGMENTATION.value


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
