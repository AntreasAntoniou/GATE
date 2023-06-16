import pytest
import torch
from torch.autograd import Variable

from gate.models.task_adapters.semantic_segmentation import (
    SimpleSegmentationDecoder,
)


def generate_input_data(bs, c, h, w):
    return [Variable(torch.randn(bs, ci, h, w)) for ci in c]


def test_simple_segmentation_decoder_normal_input():
    # Test parameters
    bs = 2
    input_feature_maps = [3, 4, 5]
    input_hw = [(8, 8), (16, 16), (32, 32)]
    num_classes = 10
    target_size = (64, 64)
    hidden_size = 16

    # Create input data
    input_list = [
        torch.randn(
            (bs, input_feature_maps[i], input_hw[i][0], input_hw[i][1])
        )
        for i in range(len(input_feature_maps))
    ]

    # Initialize the model
    model = SimpleSegmentationDecoder(
        input_list, num_classes, target_size, hidden_size
    )

    # Execute the forward pass
    output = model.forward(input_list)

    # Check the output size
    assert output.shape == (bs, num_classes, *target_size)


def test_simple_segmentation_decoder_sequence_input():
    bs = 2
    num_classes = 10
    target_size = (64, 64)
    hidden_size = 16
    sequence_input_feature_maps = [9, 16, 25]
    sequence_features = 64

    # Test with input in (b, sequence, features) format
    sequence_input_list = [
        torch.randn((bs, sequence_input_feature_maps[i], sequence_features))
        for i in range(len(sequence_input_feature_maps))
    ]

    # Initialize the model with sequence input
    sequence_model = SimpleSegmentationDecoder(
        sequence_input_list, num_classes, target_size, hidden_size
    )

    # Execute the forward pass with sequence input
    sequence_output = sequence_model.forward(sequence_input_list)

    # Check the sequence output size
    assert sequence_output.shape == (bs, num_classes, *target_size)
