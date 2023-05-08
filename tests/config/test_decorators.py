import pytest
from hydra.core.config_store import ConfigStore

import gate
from gate.boilerplate.decorators import configurable, register_configurables


def test_configurable_and_register_configurables():
    # Register the configurables
    config_store = ConfigStore.instance()

    config_store = register_configurables("gate", config_store)

    # Retrieve the configuration from the config store
    config = config_store.load("test_group/test_function")

    # Check if the configuration is loaded correctly
    assert config["test_function"]["a"] == 1
    assert config["test_function"]["b"] == 2

    # Create an instance of the configurable function with the configuration
    configured_function = test_function.__config__(**config["test_function"])

    # Check if the function is executed correctly
    assert configured_function() == 3
