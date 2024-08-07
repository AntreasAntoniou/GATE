import pytest
import torch
import torch.nn as nn

from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)


def test_gate_model_unsupported_transformation():
    source_config1 = SourceModalityConfig(text=True, image=True)
    target_config = TargetModalityConfig(image=[source_config1])

    model = GATEModel(target_config, nn.Linear(10, 10))

    with pytest.raises(ValueError, match="Unsupported modality"):
        model.process_modalities(
            "image", dict(audio=torch.randn(1, 10), image=torch.randn(1, 10))
        )
