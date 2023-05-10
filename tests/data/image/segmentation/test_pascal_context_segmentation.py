import pytest

from gate.data.image.segmentation.pascal_context import (
    build_pascal_context_dataset,
)


def test_build_pascal_context_dataset():
    # Test with download=False (Assumes the dataset is already present)
    train_data = build_pascal_context_dataset("train", download=True)
    val_data = build_pascal_context_dataset("val", download=True)
    test_data = build_pascal_context_dataset("test", download=True)

    # Check if the datasets are created without errors
    assert train_data is not None
    assert val_data is not None
    assert test_data is not None

    # Check if the datasets have the correct number of samples
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0


if __name__ == "__main__":
    pytest.main([__file__])
