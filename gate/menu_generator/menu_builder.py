import logging
import random
from typing import Callable, Dict, List, Tuple, Union

import fire
from rich import print
from rich.logging import RichHandler

# Importing various experiment commands
from gate.menu_generator.few_shot_learning_command import (
    get_commands as get_few_shot_learning_commands,
)
from gate.menu_generator.image_classification_command import (
    get_commands as get_image_classification_commands,
)
from gate.menu_generator.medical_image_classification_command import (
    get_commands as get_medical_image_classification_commands,
)
from gate.menu_generator.relational_reasoning_command import (
    get_commands as get_relational_reasoning_commands,
)
from gate.menu_generator.zero_shot_learning_command import (
    get_commands as get_zero_shot_learning_commands,
)

# Logging configuration using Rich for better terminal output
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler: RichHandler = RichHandler(markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


def run_experiments(
    prefix: str = "debug", experiment_type: str = "all"
) -> None:
    """
    Run selected or all experiments based on the argument 'experiment_type'.

    Parameters:
    - prefix (str): Prefix to identify the experiment batch. Defaults to 'debug'.
    - experiment_type (str): Type of experiment to run. Options are "zero-shot", "few-shot", "medical-class", "image-class", "relational-reasoning", and "all". Defaults to 'all'.

    Returns:
    None
    """

    # Mapping experiment types to their corresponding function calls
    experiment_funcs: Dict[str, Callable] = {
        "zero-shot": get_zero_shot_learning_commands,
        "few-shot": get_few_shot_learning_commands,
        "medical-class": get_medical_image_classification_commands,
        "image-class": get_image_classification_commands,
        "relational-reasoning": get_relational_reasoning_commands,
    }

    # Initialize an empty dictionary to hold the experiments
    experiment_dict: Dict[str, str] = {}

    # If 'all' is selected, run all types of experiments
    if experiment_type == "all":
        for exp_type, func in experiment_funcs.items():
            experiment_dict.update(func(prefix=prefix))
    else:
        # Run only the selected type of experiment
        if experiment_type in experiment_funcs:
            experiment_dict = experiment_funcs[experiment_type](prefix=prefix)
        else:
            print("Invalid experiment type selected.")
            return

    # Shuffle and run the experiments
    shuffled_experiments: List[Tuple[str, str]] = list(experiment_dict.items())
    random.shuffle(shuffled_experiments)
    for experiment_name, experiment_command in shuffled_experiments:
        print(f"Command for {experiment_name}: {experiment_command}")


# Use Google Fire for command-line argument parsing
if __name__ == "__main__":
    fire.Fire(run_experiments)
