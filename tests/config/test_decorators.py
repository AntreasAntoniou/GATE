from hydra_zen import instantiate
from numpy import insert
import pytest
from hydra.core.config_store import ConfigStore

import gate
from gate.boilerplate.decorators import configurable, register_configurables


def test_configurable_and_register_configurables():
    # Register the configurables
    config_store = ConfigStore.instance()

    config_store = register_configurables("gate", config_store)

    # Retrieve the configuration from the config store
    config = config_store.load(config_path="test_group/test_function")

    # Check if the configuration is loaded correctly
    assert config.a == 1
    assert config.b == 2

    # Create an instance of the configurable function with the configuration
    configured_function = instantiate(config)

    # Check if the function is executed correctly
    assert configured_function() == 3
