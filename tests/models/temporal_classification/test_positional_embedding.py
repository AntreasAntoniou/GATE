import pytest
import torch

from gate.models.task_adapters.temporal_image_classification import \
    PositionalEncoding


def test_output_shape():
    batch_size, seq_len, d_model = 32, 50, 512
    x = torch.randn(batch_size, seq_len, d_model)
    pos_encoding = PositionalEncoding()
    output = pos_encoding(x)
    assert (
        output.shape == x.shape
    ), f"Expected output shape {x.shape}, got {output.shape}"


def test_fixed_length_caching():
    batch_size, seq_len, d_model = 32, 50, 512
    x = torch.randn(batch_size, seq_len, d_model)

    pos_encoding = PositionalEncoding(has_fixed_context_length=True)
    output1 = pos_encoding(x)

    # Change the values in x to make sure cached positional encoding is actually being used
    x_new = torch.randn_like(x)
    output2 = pos_encoding(x_new)

    # Check if the difference between the two outputs is exactly equal to the difference between the two inputs
    # This would mean that the same positional encoding was added to both
    assert torch.allclose(
        output2 - output1, x_new - x
    ), "Cached positional encoding not reused"


@pytest.mark.parametrize("seq_len", [25, 50, 100])
def test_variable_length(seq_len):
    batch_size, d_model = 32, 512
    x = torch.randn(batch_size, seq_len, d_model)

    pos_encoding = PositionalEncoding()
    output = pos_encoding(x)

    assert (
        output.shape == x.shape
    ), f"Expected output shape {x.shape}, got {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__])
