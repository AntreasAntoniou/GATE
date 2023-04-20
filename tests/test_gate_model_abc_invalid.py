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

    with pytest.raises(ValueError, match="Unsupported transformation"):
        model.process_modalities(
            "image", audio=torch.randn(1, 10), image=torch.randn(1, 10)
        )


if __name__ == "__main__":
    test_gate_model_unsupported_transformation()
    print("All tests passed!")
