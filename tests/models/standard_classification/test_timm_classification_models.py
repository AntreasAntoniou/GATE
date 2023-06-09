import torch
import torch.nn.functional as F

from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_specific_models.classification.timm import (
    ModelAndTransform,
    build_model,
)


def test_build_model():
    model_and_transform = build_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert model_and_transform.model is not None
    assert model_and_transform.transform is not None


def test_clip_with_linear_forward():
    model_and_transform = build_model(
        clip_model_name="openai/clip-vit-base-patch32",
        timm_model_name="resnet50.a1_in1k",
        modality="image",
        num_classes=100,
    )

    x_dummy = torch.rand(2, 3, 224, 224)
    y_dummy = torch.randint(0, 100, (2,))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": x_dummy, "labels": y_dummy})

    output = model.forward(input_dict)

    assert output.shape == (2, 100)

    loss = F.cross_entropy(output, y_dummy)
    assert loss.item() > 0


def test_clip_with_linear_forward_loss():
    model_and_transform = build_model(
        clip_model_name="openai/clip-vit-base-patch32",
        timm_model_name="resnet50.a1_in1k",
        modality="image",
        num_classes=100,
    )
    model = model_and_transform.model
    transform = model_and_transform.transform

    source_config1 = SourceModalityConfig(image=True)

    target_config = TargetModalityConfig(image=[source_config1])

    x_dummy = torch.rand(2, 3, 224, 224)
    y_dummy = torch.randint(0, 100, (2,))

    model = model_and_transform.model
    transform = model_and_transform.transform

    model = GATEModel(
        target_config, model, key_remapper_dict={"image": "image"}
    )

    input_dict = transform({"image": x_dummy, "labels": y_dummy})

    output = model.forward(input_dict)
    assert output["image"]["image"].shape == (2, 100)

    loss = F.cross_entropy(output["image"]["image"], y_dummy)
    assert loss.item() > 0


if __name__ == "__main__":
    test_build_model()
    test_clip_with_linear_forward()
    test_clip_with_linear_forward_loss()
