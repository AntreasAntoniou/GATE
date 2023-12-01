from gate.menu.core import (
    AdapterTypeNames,
    DatasetName,
    EvaluatorName,
    TrainerName,
    get_model_selection,
)

dataset_configs = [DatasetName.VISUAL_RELATIONAL_REASONING.value.CLEVR]


dataset_configs = {
    dataset_name: dataset_name.value for dataset_name in dataset_configs
}

BATCH_SIZE = 64
ENCODER_CONFIG = AdapterTypeNames.RELATIONAL_REASONING.value
RESNET_LR = 1e-3
VIT_LR = 1e-5
TRAINER_NAME = TrainerName.VISUAL_RELATIONAL_REASONING.value
EVALUATOR_NAME = EvaluatorName.VISUAL_RELATIONAL_REASONING.value


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
