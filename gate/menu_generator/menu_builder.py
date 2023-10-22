import logging
import random
from typing import Callable, Dict, List, Tuple, Union

import fire
from rich import print
from rich.logging import RichHandler

from gate.menu_generator.configs.few_shot_learning import (
    config as few_shot_learning_config,
)
from gate.menu_generator.configs.image_classification import (
    config as image_classification_config,
)
from gate.menu_generator.configs.medical_image_classification import (
    config as medical_image_classification_config,
)
from gate.menu_generator.utils import build_command, get_commands

# Logging configuration using Rich for better terminal output
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler: RichHandler = RichHandler(markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# Let's adapt the experiment generator script to work with the new configuration setup.
# We'll modify the `generate_commands` and `get_commands` functions to use the new `dataset_configs` and `model_configs`.


def generate_commands(
    prefix: str,
    seed_list: List[int],
    experiment_config: Dict,
    accelerate_launch_path: str = "/opt/conda/envs/main/bin/accelerate-launch",
    gate_run_path: str = "/app/gate/run.py",
) -> Dict[str, str]:
    """
    Generate a list of commands for different experiments. 📚
    """
    dataset_dict = experiment_config["dataset"]
    model_configs = experiment_config["model"]
    trainer = experiment_config["trainer"]
    evaluator = experiment_config["evaluator"]
    command_dict = {}

    for dataset_key, dataset_value in dataset_dict.items():
        for model_key, model_config in model_configs.items():
            for seed in seed_list:
                exp_name = (
                    f"{prefix}-{dataset_value}-{model_key}-{seed}".replace(
                        "_", "-"
                    )
                )
                model_args = ""
                if model_config.encoder_config.value.timm_model_name:
                    model_args = f"model.timm_model_name={model_config.encoder_config.value.timm_model_name}"
                lr_list = model_config.learning_rate_config.get_lr()
                for lr in lr_list:
                    command = build_command(
                        exp_name=exp_name,
                        model_name=model_config.model_type,
                        dataset_name=dataset_value,
                        model_args=model_args,
                        lr=lr,
                        trainer=trainer,
                        evaluator=evaluator,
                        seed=seed,
                        train_batch_size=model_config.train_batch_size,
                        eval_batch_size=model_config.eval_batch_size,
                        accelerate_launch_path=accelerate_launch_path,
                        gate_run_path=gate_run_path,
                    )
                    command_dict[exp_name] = command
    return command_dict


def run_experiments(
    prefix: str = "debug",
    experiment_type: str = "all",
    accelerate_launch_path: str = "accelerate launch",  # "/opt/conda/envs/main/bin/accelerate-launch",
    gate_run_path: str = "gate/run.py",
) -> None:
    """
    Run selected or all experiments based on the argument 'experiment_type'.
    """
    seed_list = [7]
    experiment_dict = {}

    experiment_configs: Dict[str, Dict] = {
        "image-class": image_classification_config,
        "few-shot": few_shot_learning_config,
        "med-class": medical_image_classification_config,
    }

    if experiment_type == "all":
        for config in experiment_configs.values():
            experiment_dict.update(
                generate_commands(
                    prefix=prefix,
                    seed_list=seed_list,
                    experiment_config=config,
                    accelerate_launch_path=accelerate_launch_path,
                    gate_run_path=gate_run_path,
                )
            )
    else:
        if experiment_type in experiment_configs:
            experiment_dict = generate_commands(
                prefix=prefix,
                seed_list=seed_list,
                experiment_config=experiment_configs[experiment_type],
                accelerate_launch_path=accelerate_launch_path,
                gate_run_path=gate_run_path,
            )
        else:
            print("Invalid experiment type selected.")
            return

    for experiment_name, experiment_command in experiment_dict.items():
        print(f"{experiment_command} \n")


# Use Google Fire for command-line argument parsing
if __name__ == "__main__":
    fire.Fire(run_experiments)
