import numpy as np
import pytest

from gate.metrics.vqa_eval import vqa_metric

# Sample data
answer_list = [
    ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", "yes", "yes"],
    ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", "yes", "yes"],
    [
        "apple",
        "orange",
        "banana",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
    ],
    [
        "apple",
        "orange",
        "banana",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
    ],
    [
        "apple",
        "orange",
        "banana",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
        "apple",
    ],
]

# Test cases for predictions
prediction_dict = {
    "yes": {"overall": [1.0]},
    "no": {"overall": [0.3]},
    "apple": {"overall": [1.0]},
    "orange": {"overall": [0.3]},
    "bike": {"overall": [0.0]},
}


def test_vqa_metric():
    # Get the answers and predicted answers
    result = vqa_metric(
        answers=answer_list, predicted_answers=list(prediction_dict.keys())
    )

    correct = [item["overall"][0] for key, item in prediction_dict.items()]

    assert np.allclose(result["overall"], correct)
