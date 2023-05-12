import pytest
from gate.models import ModelAndTransform
from gate.models.core import GATEModel
from gate.models.classification.tali import (
    build_model,
    build_gate_tali_model,
)


def test_build_model():
    model_and_transform = build_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert callable(model_and_transform.transform)
    assert model_and_transform.model is not None

    # Ensure the function raises an error for unsupported modalities
    with pytest.raises(ValueError):
        build_model(modality="unsupported_modality")


def test_build_gate_tali_model():
    model_and_transform = build_gate_tali_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert callable(model_and_transform.transform)
    assert isinstance(model_and_transform.model, GATEModel)

    # Ensure the function raises an error for unsupported modalities
    with pytest.raises(ValueError):
        build_gate_tali_model(modality="unsupported_modality")
