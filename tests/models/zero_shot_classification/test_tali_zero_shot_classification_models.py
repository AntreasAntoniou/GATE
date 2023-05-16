import pytest
import torch

from gate.models.task_specific_models.zero_shot_classification.tali import (
    ModelAndTransform,
    build_model,
    build_gate_model,
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
    )

    image = torch.rand(20, 3, 224, 224)
    text = ["Let's go for a walk"] * 20

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": image, "text": text})

    output = model.forward(**input_dict, return_loss=True)

    output["loss"].backward()

    # assert output["logits"].shape == (2, 10, 5)

    # assert output["loss"].item() > 0


@pytest.mark.parametrize(
    "pretrained,num_projection_features", pytest_parameters
)
def test_model_gate_with_linear_forward(pretrained, num_projection_features):
    model_and_transform = build_gate_model(
        modality_a_identifier="image",
        modality_b_identifier="text",
        pretrained=pretrained,
        num_projection_features=num_projection_features,
    )

    image = torch.rand(20, 3, 224, 224)
    text = ["Let's go for a walk"] * 20

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": image, "text": text})

    output = model.forward(input_dict, return_loss=True)

    output["image_text"]["image_text"]["loss"].backward()

    # assert output["image"]["image"]["logits"].shape == (2, 10, 5)

    # assert output["image"]["image"]["loss"].item() > 0


if __name__ == "__main__":
    test_build_model()
    test_model_gate_with_linear_forward()
