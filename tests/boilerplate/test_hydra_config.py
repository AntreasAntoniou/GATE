import pytest
from hydra.experimental import compose, initialize

from gate.boilerplate.utils import pretty_config
from gate.config import BaseConfig, collect_config_store


def test_collect_config_store():
    from rich import print

    # Collect and store the configurations
    config_store = collect_config_store()

    with initialize(config_path=None, job_name="config"):
        # Compose the configuration
        cfg = compose()

        # Print the configuration
        print(pretty_config(cfg, resolve=True))
