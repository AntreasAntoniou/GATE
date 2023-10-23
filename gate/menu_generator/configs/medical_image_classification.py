from gate.menu_generator.core import (
    DatasetName,
    EncoderNames,
    EvaluatorName,
    LearningRateConfig,
    ModelConfig,
    ModelTypeNames,
    TrainerName,
)

dataset_configs = [DatasetName.CHX, DatasetName.HAM10K, DatasetName.DR]


dataset_configs = {
    dataset_name: dataset_name.value for dataset_name in dataset_configs
}

BATCH_SIZE = 64
MODEL_TYPE = ModelTypeNames.TIMM_IMAGE_CLASSIFICATION.value
RESNET_LR = 1e-3
VIT_LR = 1e-5
TRAINER_NAME = TrainerName.MULTI_CLASS_CLASSIFICATION.value
EVALUATOR_NAME = EvaluatorName.MULTI_CLASS_CLASSIFICATION.value
model_configs = {
    EncoderNames.CLIPViTBase16_224.value.pretty_name: ModelConfig(
        model_type=MODEL_TYPE,
        encoder_config=EncoderNames.CLIPViTBase16_224,
        learning_rate_config=LearningRateConfig(
            default=[VIT_LR], dataset_specific={}
        ),
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    ),
    # EncoderNames.LaionViTBase16_224.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.LaionViTBase16_224,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    EncoderNames.ResNet50A1.value.pretty_name: ModelConfig(
        model_type=MODEL_TYPE,
        encoder_config=EncoderNames.ResNet50A1,
        learning_rate_config=LearningRateConfig(
            default=[RESNET_LR], dataset_specific={}
        ),
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    ),
    # EncoderNames.SamViTBase16_224.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.SamViTBase16_224,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    # EncoderNames.AugRegViTBase16_224.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.AugRegViTBase16_224,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    # EncoderNames.DiNoViTBase16_224.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.DiNoViTBase16_224,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    # EncoderNames.EffNetV2_RW_S_RA2.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.EffNetV2_RW_S_RA2,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    # EncoderNames.DeiT3BasePatch16_224.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.DeiT3BasePatch16_224,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    # EncoderNames.ResNeXt50_32x4dA1.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.ResNeXt50_32x4dA1,
    #     learning_rate_config=LearningRateConfig(
    #         default=[RESNET_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    # EncoderNames.FlexViTBase_1200EP.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.FlexViTBase_1200EP,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
    # EncoderNames.IJEPAViTHugePatch14_224.value.pretty_name: ModelConfig(
    #     model_type=MODEL_TYPE,
    #     encoder_config=EncoderNames.IJEPAViTHugePatch14_224,
    #     learning_rate_config=LearningRateConfig(
    #         default=[VIT_LR], dataset_specific={}
    #     ),
    #     train_batch_size=BATCH_SIZE,
    #     eval_batch_size=BATCH_SIZE,
    # ),
}


config = {
    "dataset": dataset_configs,
    "model": model_configs,
    "trainer": TRAINER_NAME,
    "evaluator": EVALUATOR_NAME,
}
