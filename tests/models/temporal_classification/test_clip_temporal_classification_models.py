import pytest
import torch

from gate.models.task_specific_models.temporal_image_classification.clip import (
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


@pytest.mark.parametrize("pretrained,num_classes", pytest_parameters)
def test_model_with_linear_forward(pretrained, num_classes):
    model_and_transform = build_model(
        pretrained=pretrained,
        num_classes=num_classes,
    )

    inputs = torch.rand(2, 10, 3, 224, 224)
    labels = torch.randint(0, num_classes, (2,))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform(
        {"video": inputs, "labels": labels, "return_loss_and_metrics": True}
    )

    output = model.forward(**input_dict)

    assert output["logits"].shape == (2, 512)

    assert output["loss"].item() > 0


@pytest.mark.parametrize("pretrained,num_classes", pytest_parameters)
def test_model_gate_with_linear_forward(pretrained, num_classes):
    model_and_transform = build_gate_model(
        pretrained=pretrained,
        num_classes=num_classes,
    )

    inputs = torch.rand(2, 10, 3, 224, 224)
    labels = torch.randint(0, num_classes, (2,))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform(
        {"video": inputs, "labels": labels, "return_loss_and_metrics": True}
    )

    output = model.forward(input_dict)

    assert output["video"]["video"]["logits"].shape == (2, 512)

    assert output["video"]["video"]["loss"].item() > 0
