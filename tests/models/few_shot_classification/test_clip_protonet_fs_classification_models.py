import pytest
import torch

from gate.models.task_specific_models.few_shot_classification.clip_protonet import (
    ModelAndTransform,
    build_gate_model,
    build_model,
)

pytest_parameters = [(True, 512), (False, 512), (True, None), (False, None)]


def test_build_model():
    model_and_transform = build_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert model_and_transform.model is not None
    assert model_and_transform.transform is not None


@pytest.mark.parametrize("pretrained,num_output_features", pytest_parameters)
def test_model_with_linear_forward(pretrained, num_output_features):
    model_and_transform = build_model(
        modality="image",
        pretrained=pretrained,
        num_output_features=num_output_features,
    )

    support_set_inputs = torch.rand(2, 10, 3, 224, 224)
    query_set_inputs = torch.rand(2, 10, 3, 224, 224)
    support_set_labels = torch.randint(0, 5, (2, 10))
    query_set_labels = torch.randint(0, 5, (2, 10))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform(
        {
            "image": {
                "support_set_inputs": support_set_inputs,
                "query_set_inputs": query_set_inputs,
                "support_set_labels": support_set_labels,
                "query_set_labels": query_set_labels,
            }
        }
    )

    output = model.forward(**input_dict)

    assert output["logits"].shape == (2, 10, 5)

    assert output["loss"].item() > 0


@pytest.mark.parametrize("pretrained,num_output_features", pytest_parameters)
def test_model_gate_with_linear_forward(pretrained, num_output_features):
    model_and_transform = build_gate_model(
        modality="image",
        pretrained=pretrained,
        num_output_features=num_output_features,
    )

    support_set_inputs = torch.rand(2, 10, 3, 224, 224)
    query_set_inputs = torch.rand(2, 10, 3, 224, 224)
    support_set_labels = torch.randint(0, 5, (2, 10))
    query_set_labels: torch.Tensor = torch.randint(0, 5, (2, 10))

    model = model_and_transform.model
    transform = model_and_transform.transform

    input_dict = transform(
        {
            "image": {
                "support_set_inputs": support_set_inputs,
                "query_set_inputs": query_set_inputs,
                "support_set_labels": support_set_labels,
                "query_set_labels": query_set_labels,
            }
        }
    )

    output = model.forward(input_dict)

    assert output["image"]["image"]["logits"].shape == (2, 10, 5)

    assert output["image"]["image"]["loss"].item() > 0


if __name__ == "__main__":
    test_build_model()
    test_clip_with_linear_forward()
