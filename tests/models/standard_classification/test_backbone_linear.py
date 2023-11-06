import pytest
import torch
import torch.nn as nn

from gate.menu.core import EncoderNames
from gate.models.backbones.clip_image import CLIPVisionAdapter
from gate.models.backbones.clip_text import CLIPTextAdapter
from gate.models.backbones.timm import CLIPModelPaths, TimmCLIPAdapter
from gate.models.core import GATEModel
from gate.models.task_adapters.classification import BackboneWithLinear

data = [
    (
        TimmCLIPAdapter,
        dict(
            timm_model_name=EncoderNames.EffNetV2_RW_S_RA2.value.timm_model_name,
            clip_model_name=CLIPModelPaths.openai_b_16,
        ),
    ),  # In this example, Class1 takes two arguments, a and b.
    (
        CLIPVisionAdapter,
        dict(model_name=CLIPModelPaths.openai_b_16, image_size=224),
    ),
    (
        CLIPTextAdapter,
        dict(model_name=CLIPModelPaths.openai_b_16, image_size=224),
    ),
]


# Below, we are using pytest's parameterize feature to create a version of the
# `test_with_linear_forward_loss` test for all cases in the data list. Each run of the test will
# create an instance of the class with the provided arguments.
@pytest.mark.parametrize("encoder_class,arg_dict", data)
def test_with_linear_forward_loss(encoder_class, arg_dict):
    x_dummy = torch.rand(2, 3, 224, 224)
    y_dummy = torch.randint(0, 100, (2,))

    encoder = encoder_class(**arg_dict)
    model = BackboneWithLinear(
        encoder=encoder, pretrained=False, num_classes=100
    )
    transform = model.adapter_transforms
    model = GATEModel(config=model.modality_config, model=model)
    input_dict = transform({"image": x_dummy, "labels": y_dummy})

    output = model.forward(input_dict)
    assert output["image"]["image"]["logits"].shape == (2, 100)

    loss = output["image"]["image"]["loss"]

    assert loss.item() > 0

    loss.backward()
