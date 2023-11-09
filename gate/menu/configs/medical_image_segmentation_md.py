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

dataset_configs = DatasetName.MEDICAL_DECATHLON_SEGMENTATION.value


dataset_configs = {
    dataset_name: dataset_name.value for dataset_name in dataset_configs
}

BATCH_SIZE = 1
MODEL_TYPE = AdapterTypeNames.TIMM_MD_SEGMENTATION.value
RESNET_LR = 6e-4
VIT_LR = 6e-6
TRAINER_NAME = TrainerName.MEDICAL_SEMANTIC_SEGMENTATION.value
EVALUATOR_NAME = EvaluatorName.MEDICAL_SEMANTIC_SEGMENTATION.value


config = {
    "dataset": dataset_configs,
    "model": get_model_selection(
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        resnet_lr=RESNET_LR,
        vit_lr=VIT_LR,
        wd=0.0,
    ),
    "trainer": TRAINER_NAME,
    "evaluator": EVALUATOR_NAME,
}
