import pytest
import torch

from gate.models.task_specific_models.semantic_segmentation.clip import (
    ModelAndTransform,
    build_gate_model,
    build_model,
)

pytest_parameters = [(True), (False)]


def test_build_model():
    model_and_transform = build_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert model_and_transform.model is not None
    assert model_and_transform.transform is not None


@pytest.mark.parametrize("pretrained", pytest_parameters)
def test_model_with_linear_forward(pretrained):
    model_and_transform = build_model(
        num_classes=100,
        pretrained=pretrained,
    )

    image = torch.rand(2, 3, 224, 224)
    labels = torch.randint(low=0, high=100, size=(2, 1, 224, 224))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": image, "labels": labels})

    output = model.forward(**input_dict)

    output["loss"].backward()
    # assert output["logits"].shape == (2, 10, 5)

    # assert output["loss"].item() > 0


@pytest.mark.parametrize("pretrained", pytest_parameters)
def test_model_gate_with_linear_forward(pretrained):
    model_and_transform = build_gate_model(
        num_classes=100,
        pretrained=pretrained,
    )

    image = torch.rand(2, 3, 224, 224)
    labels = torch.randint(low=0, high=100, size=(2, 1, 224, 224))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": image, "labels": labels})

    output = model.forward(input_dict)

    output["image"]["image"]["loss"].backward()

    # assert output["image"]["image"]["logits"].shape == (2, 10, 5)

    # assert output["image"]["image"]["loss"].item() > 0


if __name__ == "__main__":
    test_build_model()
    test_model_gate_with_linear_forward()
