import torch
from rich.traceback import install
from torch import nn
from torch.testing import assert_close

install()

from gate.models.core import Ensemble


def test_ensemble():
    # Create some simple models
    model1 = nn.Linear(10, 5)
    model2 = nn.Linear(10, 5)

    # Create the ensemble
    ensemble = Ensemble([model1, model2])

    # Create a dummy input
    x = torch.randn(1, 10)

    # Pass the input through the ensemble
    output = ensemble(x)

    # Check that the output is a dictionary with the correct keys
    assert isinstance(output, dict)
    assert set(output.keys()) == {
        "ensemble_pred",
        "weighted_ensemble_pred",
        "soup_pred",
        "weighted_soup_pred",
    }

    # Check that the output values have the correct shape
    for value in output.values():
        assert value.shape == (1, 5)

    # Check that the ensemble prediction is the mean of the individual model predictions
    expected_ensemble_pred = (model1(x) + model2(x)) / 2
    assert_close(
        output["ensemble_pred"], expected_ensemble_pred, rtol=1e-5, atol=1e-8
    )

    # Check that the weighted ensemble prediction is the weighted mean of the individual model predictions
    expected_weighted_ensemble_pred = ensemble.weights[0] * model1(
        x
    ) + ensemble.weights[1] * model2(x)
    assert_close(
        output["weighted_ensemble_pred"],
        expected_weighted_ensemble_pred,
        rtol=1e-5,
        atol=1e-8,
    )

    # Check that the soup model prediction is close to the ensemble prediction
    # (they won't be exactly equal because the soup model has its own set of parameters)
    assert_close(
        output["soup_pred"], expected_ensemble_pred, rtol=1e-1, atol=1e-1
    )

    # Check that the weighted soup model prediction is close to the weighted ensemble prediction
    # (they won't be exactly equal because the weighted soup model has its own set of parameters)
    assert_close(
        output["weighted_soup_pred"],
        expected_weighted_ensemble_pred,
        rtol=1e-1,
        atol=1e-1,
    )
