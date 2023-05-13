import pathlib
import pytest
import os

import learn2learn as l2l

from gate.data.few_shot.aircraft import (
    build_aircraft_dataset,
)


@pytest.fixture
def aircraft_dataset():
    dataset_root = pathlib.Path(os.environ["PYTEST_DIR"])
    return build_aircraft_dataset(
        set_name="train", data_dir=dataset_root, num_tasks=1000000
    )


def test_aircraft_dataset_creation(aircraft_dataset):
    assert isinstance(aircraft_dataset, l2l.data.TaskDataset)


def test_aircraft_dataset_length(aircraft_dataset):
    assert len(aircraft_dataset) == 1000000


def test_aircraft_dataset_sample(aircraft_dataset):
    episode = aircraft_dataset[0]
    print(episode)
    support_set, query_set = episode

    assert len(support_set) == 2
    assert len(query_set) == 2

    support_inputs, support_targets = support_set
    query_inputs, query_targets = query_set

    assert support_inputs.shape == (5, 5, 84, 84, 3)
    assert support_targets.shape == (5, 5)
    assert query_inputs.shape == (5, 15, 84, 84, 3)
    assert query_targets.shape == (5, 15)
