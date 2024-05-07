import pytest
import torch

from gate.menu.core import Encoders
from gate.models.backbones.bart_text import BartAdapter, BartModelPaths
from gate.models.backbones.bert_text import BertAdapter, BertModelPaths
from gate.models.backbones.clip_image import CLIPVisionAdapter
from gate.models.backbones.clip_text import CLIPTextAdapter
from gate.models.backbones.mpnet_text import MPNetAdapter, MPNetModelPaths
from gate.models.backbones.timm import CLIPModelPaths, TimmCLIPAdapter
from gate.models.backbones.wave2vec_audio import (
    Wav2Vec2ModelPaths,
    Wav2VecV2Adapter,
)
from gate.models.backbones.whisper_audio import (
    WhisperAdapter,
    WhisperModelPaths,
)
from gate.models.core import GATEModel
from gate.models.adapters.zero_shot_classification import (
    DuoModalZeroShotModel,
)

data = [
    (
        TimmCLIPAdapter,
        dict(
            timm_model_name=Encoders.EffNetV2_RW_S_RA2.value.timm_model_name,
            clip_model_name=CLIPModelPaths.openai_b_16,
            num_projection_features=64,
        ),
    ),
    (
        TimmCLIPAdapter,
        dict(
            timm_model_name=Encoders.AugRegViTBase16_224.value.timm_model_name,
            clip_model_name=CLIPModelPaths.openai_b_16,
            num_projection_features=64,
        ),
    ),
    (
        CLIPVisionAdapter,
        dict(
            model_name=CLIPModelPaths.openai_b_16,
            image_size=224,
            num_projection_features=64,
        ),
    ),
    (
        CLIPTextAdapter,
        dict(
            model_name=CLIPModelPaths.openai_b_16,
            image_size=224,
            num_projection_features=64,
        ),
    ),
    (
        BertAdapter,
        dict(
            clip_model_name=CLIPModelPaths.openai_b_16,
            bert_model_name=BertModelPaths.base_uncased,
            image_size=224,
            num_projection_features=64,
        ),
    ),
    (
        MPNetAdapter,
        dict(
            clip_model_name=CLIPModelPaths.openai_b_16,
            mpnet_model_name=MPNetModelPaths.base,
            image_size=224,
            num_projection_features=64,
        ),
    ),
    (
        BartAdapter,
        dict(
            clip_model_name=CLIPModelPaths.openai_b_16,
            bart_model_name=BartModelPaths.base,
            image_size=224,
            num_projection_features=64,
        ),
    ),
    (
        WhisperAdapter,
        dict(
            clip_model_name=CLIPModelPaths.openai_b_16,
            whisper_model_name=WhisperModelPaths.base,
            image_size=224,
            num_projection_features=64,
        ),
    ),
    (
        Wav2VecV2Adapter,
        dict(
            clip_model_name=CLIPModelPaths.openai_b_16,
            wav2vec2_model_name=Wav2Vec2ModelPaths.base,
            image_size=224,
            num_projection_features=64,
        ),
    ),
]


# Below, we are using pytest's parameterize feature to create a version of the
# `test_with_linear_forward_loss` test for all cases in the data list. Each run of the test will
# create an instance of the class with the provided arguments.
@pytest.mark.parametrize("encoder_class,arg_dict", data)
def test_with_linear_forward_loss(encoder_class, arg_dict):
    image = torch.rand(20, 3, 224, 224)
    text = ["Let's go for a walk"] * 20

    encoder = encoder_class(**arg_dict)
    model = DuoModalZeroShotModel(encoder=encoder, projection_num_features=512)
    transform = model.adapter_transforms

    model = GATEModel(config=model.modality_config, model=model)

    input_dict = transform({"image": image, "text": text})

    input_dict["return_loss"] = True

    output = model.forward(input_dict)

    assert output["image_text"]["image_text"]["logits"]["similarities"][
        "image_to_text_similarities"
    ].shape == (20, 20)

    loss = output["image_text"]["image_text"]["loss"]

    assert loss.item() > 0

    loss.backward()


@pytest.mark.parametrize("encoder_class,arg_dict", data)
def test_with_linear_forward_loss_5D(encoder_class, arg_dict):
    image = torch.rand(10, 2, 3, 224, 224)
    text = [["Let's go for a walk"] * 2] * 10

    encoder = encoder_class(**arg_dict)
    model = DuoModalZeroShotModel(encoder=encoder, projection_num_features=512)
    transform = model.adapter_transforms

    model = GATEModel(config=model.modality_config, model=model)

    input_dict = transform({"image": image, "text": text})

    input_dict["return_loss"] = True

    output = model.forward(input_dict)
    assert output["image_text"]["image_text"]["logits"]["similarities"][
        "image_to_text_similarities"
    ].shape == (20, 20)

    loss = output["image_text"]["image_text"]["loss"]

    assert loss.item() > 0

    loss.backward()
