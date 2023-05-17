import pytest
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from numpy import insert

import gate
from gate.boilerplate.decorators import configurable, register_configurables


def test_configurable_and_register_configurables():
    # Register the configurables
    config_store = ConfigStore.instance()

    config_store = register_configurables("gate", config_store)

    # Retrieve the configuration from the config store
    config = config_store.load(config_path="test_group/test_function.yaml")

    # Check if the configuration is loaded correctly
    # ConfigNode(name='test_function.yaml',
    # node={'_target_': 'gate.test_module.test_function', 'a': 1, 'b': 2},
    # group='test_group', package=None, provider=None)
    assert config.node["a"] == 1
    assert config.node["b"] == 2

    # Create an instance of the configurable function with the configuration
    configured_function = instantiate(config.node)

    # Check if the function is executed correctly
    assert configured_function == 3
