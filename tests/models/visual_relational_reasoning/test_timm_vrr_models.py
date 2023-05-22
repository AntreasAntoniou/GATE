import pytest
import torch

from gate.models.task_specific_models.visual_relational_reasoning.timm import (
    ModelAndTransform,
    build_gate_model,
    build_model,
)

pytest_parameters = [(True, 512), (False, 512)]


def test_build_model():
    model_and_transform = build_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert model_and_transform.model is not None
    assert model_and_transform.transform is not None


@pytest.mark.parametrize(
    "pretrained,num_projection_features", pytest_parameters
)
def test_model_with_linear_forward(pretrained, num_projection_features):
    model_and_transform = build_model(
        modality_a_identifier="image",
        modality_b_identifier="text",
        pretrained=pretrained,
        num_projection_features=num_projection_features,
        num_classes=10,
    )

    image = torch.rand(20, 3, 224, 224)
    text = ["Let's go for a walk"] * 20
    labels = torch.randint(0, 10, (20,))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": image, "text": text, "labels": labels})

    output = model.forward(**input_dict)

    output["loss"].backward()


@pytest.mark.parametrize(
    "pretrained,num_projection_features", pytest_parameters
)
def test_model_gate_with_linear_forward(pretrained, num_projection_features):
    model_and_transform = build_gate_model(
        modality_a_identifier="image",
        modality_b_identifier="text",
        pretrained=pretrained,
        num_projection_features=num_projection_features,
        num_classes=10,
    )

    image = torch.rand(20, 3, 224, 224)
    text = ["Let's go for a walk"] * 20
    labels = torch.randint(0, 10, (20,))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": image, "text": text, "labels": labels})

    output = model.forward(input_dict)

    output["image_text"]["image_text"]["loss"].backward()


if __name__ == "__main__":
    test_build_model()
    test_model_gate_with_linear_forward()
