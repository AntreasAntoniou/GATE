# gate/example_module.py
from gate.boilerplate.decorators import configurable


@configurable(
    group="test_group", name="test_function", defaults={"a": 1, "b": 2}
)
def test_function(a: int, b: int) -> int:
    return a + b
