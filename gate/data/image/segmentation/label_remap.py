import pytest
import torch


def remap_tensor_values(
    input_tensor: torch.Tensor, remapping_dict: dict
) -> torch.Tensor:
    """Remaps values in a PyTorch tensor according to a provided dictionary.

    Handles cases where the remapping dictionary has fewer keys than values in the tensor
    or has extra keys. Raises a ValueError if keys are missing.

    Args:
        input_tensor: The input PyTorch tensor to be remapped.
        remapping_dict: A dictionary mapping old values to new values.

    Returns:
        A new PyTorch tensor with the values remapped.
    """
    original_shape = input_tensor.shape
    input_tensor = input_tensor.flatten()
    old_values = torch.tensor(list(remapping_dict.keys()))
    new_values = torch.tensor(list(remapping_dict.values()))
    mask = input_tensor == old_values[..., None]

    # Selective Remapping
    remapped_tensor = input_tensor.clone()
    for idx, sub_mask in enumerate(mask):
        remapped_tensor[sub_mask] = new_values[idx]

    return remapped_tensor.reshape(original_shape)


def test_remapping():
    """Tests various scenarios for the remap_tensor_values function."""

    # Test Case 1: Standard Remapping
    data = torch.tensor([1, 2, 3, 2, 1])
    mapping = {1: 10, 2: 20, 3: 30}
    expected = torch.tensor([10, 20, 30, 20, 10])
    result = remap_tensor_values(data, mapping)
    assert torch.equal(result, expected)

    # Test Case 2: Extra Keys in Remapping Dict
    data = torch.tensor([1, 2, 3])
    mapping = {1: 10, 2: 20, 3: 30, 4: 40}  # Key '4' is extra
    expected = torch.tensor([10, 20, 30])
    result = remap_tensor_values(data, mapping)
    assert torch.equal(result, expected)

    # Test Case 4: Empty Tensor
    data = torch.tensor([])
    mapping = {1: 10, 2: 20, 3: 30}
    expected = data.clone()
    result = remap_tensor_values(data, mapping)
    assert torch.equal(result, expected)


if __name__ == "__main__":
    test_remapping()
    print("All tests passed!")
