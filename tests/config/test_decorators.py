import pytest
from hydra_zen import ConfigStore
import gate
from gate.boilerplate.decorators import configurable, register_configurables


# Define a sample configurable function
@configurable(
    group="test_group", name="test_function", defaults={"a": 1, "b": 2}
)
def test_function(a: int, b: int) -> int:
    return a + b


# Register the test_function in the your_module
gate.test_function = test_function


def test_configurable_and_register_configurables():
    # Register the configurables
    register_configurables("your_module")

    # Retrieve the configuration from the config store
    config_store = ConfigStore.instance()
    config = config_store.load(group="test_group", name="test_function")

    # Check if the configuration is loaded correctly
    assert config["test_function"]["a"] == 1
    assert config["test_function"]["b"] == 2

    # Create an instance of the configurable function with the configuration
    configured_function = test_function.__config__(**config["test_function"])

    # Check if the function is executed correctly
    assert configured_function() == 3
