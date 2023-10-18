import torch
import torch.nn.functional as F

from gate.models.task_specific_models.classification.clip import (
    ModelAndTransform,
    build_gate_model,
    build_model,
)


def test_build_model():
    model_and_transform = build_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert model_and_transform.model is not None
    assert model_and_transform.transform is not None


def test_with_linear_forward_loss():
    x_dummy = torch.rand(2, 3, 224, 224)
    y_dummy = torch.randint(0, 100, (2,))

    model_and_transform = build_gate_model(pretrained=False, num_classes=100)
    model = model_and_transform.model
    transform = model_and_transform.transform
    input_dict = transform({"image": x_dummy, "labels": y_dummy})

    output = model.forward(input_dict)
    assert output["image"]["image"]["logits"].shape == (2, 100)

    loss = output["image"]["image"]["loss"]
    assert loss.item() > 0

    loss.backward()
