import torch.nn as nn

from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)


def test_gate_model_initialization_and_valid_combinations():
    source_config1 = SourceModalityConfig(text=True, image=True)
    source_config2 = SourceModalityConfig(audio=True, image=True)
    source_config3 = SourceModalityConfig(audio=True, text=True, image=True)

    target_config = TargetModalityConfig(
        image=[source_config1, source_config2, source_config3]
    )

    model = GATEModel(target_config, nn.Linear(10, 10))

    valid_combinations = model.get_valid_combinations()

    expected_combinations = [
        (("image", "text"), "image"),
        (("image", "audio"), "image"),
        (("image", "text", "audio"), "image"),
    ]

    assert len(valid_combinations) == len(expected_combinations)
    for combination in expected_combinations:
        assert combination in valid_combinations


if __name__ == "__main__":
    test_gate_model_initialization_and_valid_combinations()
    print("All tests passed!")
