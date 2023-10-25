import logging
import random
import subprocess
from typing import Callable, Dict, List, Optional, Tuple, Union

import fire
from rich import print
from rich.logging import RichHandler

from gate.menu.configs.few_shot_learning import (
    config as few_shot_learning_config,
)
from gate.menu.configs.image_classification import (
    config as image_classification_config,
)
from gate.menu.configs.image_segmentation import (
    config as image_segmentation_config,
)
from gate.menu.configs.image_text_zero_shot_classification import (
    config as image_text_zero_shot_classification_config,
)
from gate.menu.configs.medical_image_classification import (
    config as medical_image_classification_config,
)
from gate.menu.configs.medical_image_segmentation_acdc import (
    config as acdc_config,
)
from gate.menu.configs.medical_image_segmentation_md import config as md_config
from gate.menu.configs.relational_reasoning import config as rr_config
from gate.menu.configs.relational_reasoning_mm import config as rr_mm_config
from gate.menu.configs.video_classification import (
    config as video_classification_config,
)
from gate.menu.utils import build_command, get_commands

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
    num_workers: int = 12,
    gpu_ids: Optional[Union[str, int]] = None,
    accelerate_launch_path: str = "/opt/conda/envs/main/bin/accelerate-launch",
    gate_run_path: str = "/app/gate/run.py",
) -> Dict[str, str]:
    """
    Generate a dictionary of experiment commands based on the given prefix, seed list, experiment configuration, and other parameters.

    Args:
        prefix (str): Prefix for the experiment name.
        seed_list (List[int]): List of seed values for experiments.
        experiment_config (Dict): Configuration dictionary containing dataset, model, trainer, and evaluator information.
        num_workers (int, optional): Number of workers to use for running experiments. Defaults to 12.
        accelerate_launch_path (str, optional): Path to the accelerate launch script. Defaults to "/opt/conda/envs/main/bin/accelerate-launch".
        gate_run_path (str, optional): Path to the GATE run script. Defaults to "/app/gate/run.py".

    Returns:
        Dict[str, str]: A dictionary containing experiment names as keys and the corresponding experiment commands as values.
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
                        num_workers=num_workers,
                        gpu_ids=gpu_ids,
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
    accelerate_launch_path: str = "accelerate launch",
    gate_run_path: str = "gate/run.py",
    num_workers: int = 12,
    gpu_ids: Optional[Union[str, int]] = None,
    print_commands: bool = True,
    run_commands: bool = False,
) -> None:
    """
    Run selected or all experiments based on the argument 'experiment_type'.

    Args:
        prefix (str): Prefix for the experiment name.
        experiment_type (str): Type of experiment to run. Can be 'all', 'image-class', 'few-shot', or 'med-class'.
        accelerate_launch_path (str): Path to the accelerate launch script.
        gate_run_path (str): Path to the GATE run script.
        num_workers (int): Number of workers to use for running experiments.
        print_commands (bool): Whether to print the experiment commands.
        run_commands (bool): Whether to run the experiment commands.

    Returns:
        experiment_dict (dict): A dictionary containing the experiment names as keys and the corresponding experiment commands as values.
    """
    seed_list = [7]
    experiment_dict = {}

    experiment_configs: Dict[str, Dict] = {
        "image-class": image_classification_config,
        "few-shot": few_shot_learning_config,
        "med-class": medical_image_classification_config,
        "image-seg": image_segmentation_config,
        "image-text": image_text_zero_shot_classification_config,
        "acdc": acdc_config,
        "md": md_config,
        "rr": rr_config,
        "rr-mm": rr_mm_config,
        "video-class": video_classification_config,
    }

    if experiment_type == "all":
        for config in experiment_configs.values():
            experiment_dict.update(
                generate_commands(
                    prefix=prefix,
                    seed_list=seed_list,
                    experiment_config=config,
                    num_workers=num_workers,
                    accelerate_launch_path=accelerate_launch_path,
                    gate_run_path=gate_run_path,
                    gpu_ids=gpu_ids,
                )
            )
    else:
        if experiment_type in experiment_configs:
            experiment_dict = generate_commands(
                prefix=prefix,
                seed_list=seed_list,
                experiment_config=experiment_configs[experiment_type],
                num_workers=num_workers,
                accelerate_launch_path=accelerate_launch_path,
                gate_run_path=gate_run_path,
                gpu_ids=gpu_ids,
            )
        else:
            print("Invalid experiment type selected.")
            return

    if print_commands:
        for experiment_name, experiment_command in experiment_dict.items():
            print(f"Running: {experiment_command}")

            if run_commands:
                # Execute the command and capture stdout and stderr
                process = subprocess.Popen(
                    experiment_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Print stdout and stderr in real-time
                for line in iter(process.stdout.readline, b""):
                    print(line.decode().strip())

                for line in iter(process.stderr.readline, b""):
                    print(line.decode().strip())

                # Wait for the process to complete and get the exit code
                process.communicate()
                exit_code = process.returncode

                if exit_code != 0:
                    print(
                        f"Error executing {experiment_name}. Continuing with the next command."
                    )

    return experiment_dict


# Use Google Fire for command-line argument parsing
if __name__ == "__main__":
    fire.Fire(run_experiments)
