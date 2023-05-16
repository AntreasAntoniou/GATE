import pytest
import torch

from gate.models.task_specific_models.temporal_image_classification.tali_temporal_image_classification import (
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


@pytest.mark.parametrize("pretrained,num_classes", pytest_parameters)
def test_model_with_linear_forward(pretrained, num_classes):
    model_and_transform = build_model(
        modality="image",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    inputs = torch.rand(2, 10, 3, 224, 224)
    labels = torch.randint(0, num_classes, (2, 10))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": inputs})

    output = model.forward(**input_dict)

    # assert output["logits"].shape == (2, 10, 5)

    # assert output["loss"].item() > 0


@pytest.mark.parametrize("pretrained,num_classes", pytest_parameters)
def test_model_gate_with_linear_forward(pretrained, num_classes):
    model_and_transform = build_gate_model(
        modality="image",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    inputs = torch.rand(2, 10, 3, 224, 224)
    labels = torch.randint(0, num_classes, (2, 10))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform({"image": inputs})

    output = model.forward(input_dict)

    # assert output["image"]["image"]["logits"].shape == (2, 10, 5)

    # assert output["image"]["image"]["loss"].item() > 0


if __name__ == "__main__":
    test_build_model()
    test_model_gate_with_linear_forward()
