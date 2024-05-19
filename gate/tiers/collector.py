import json
import logging
import random
import subprocess
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

import fire
from rich import print
from rich.logging import RichHandler

from gate.tiers.configs.few_shot_learning import (
    Config as few_shot_learning_config,
)
from gate.tiers.configs.image_classification import (
    Config as image_classification_config,
)
from gate.tiers.configs.image_segmentation import (
    Config as image_segmentation_config,
)
from gate.tiers.configs.image_text_zero_shot_classification import (
    Config as image_text_zero_shot_classification_config,
)
from gate.tiers.configs.medical_image_classification import (
    Config as medical_image_classification_config,
)
from gate.tiers.configs.medical_image_segmentation_acdc import (
    Config as acdc_config,
)
from gate.tiers.configs.medical_image_segmentation_md import (
    Config as md_config,
)
from gate.tiers.configs.relational_reasoning import Config as rr_config
from gate.tiers.configs.relational_reasoning_mm import Config as rr_mm_config
from gate.tiers.configs.video_classification import (
    Config as video_classification_config,
)
from gate.tiers.utils import build_command

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
    train_iters: int = 10000,
    evaluate_every_n_steps: int = 250,
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
                encoder_args = ""
                for key, value in asdict(model_config.encoder_config).items():
                    if "pretty_name" in key:
                        continue
                    if "encoder_name" in key:
                        continue
                    if value is None:
                        continue

                    encoder_args += f"encoder.{key}={value} "

                adapter_args = ""
                for key, value in asdict(model_config.adapter_config).items():
                    if "pretty_name" in key:
                        continue
                    if "adapter_name" in key:
                        continue
                    if value is None:
                        continue

                    adapter_args += f"adapter.{key}={value} "

                lr_list = model_config.learning_rate_config.get_lr()
                for lr in lr_list:
                    command = build_command(
                        exp_name=exp_name,
                        encoder_name=model_config.encoder_config.encoder_name,
                        adapter_name=model_config.adapter_config.adapter_name,
                        dataset_name=dataset_value,
                        encoder_args=encoder_args,
                        adapter_args=adapter_args,
                        num_workers=num_workers,
                        gpu_ids=gpu_ids,
                        lr=lr,
                        weight_decay=model_config.weight_decay,
                        trainer=trainer,
                        evaluator=evaluator,
                        seed=seed,
                        train_batch_size=model_config.train_batch_size,
                        eval_batch_size=model_config.eval_batch_size,
                        accelerate_launch_path=accelerate_launch_path,
                        gate_run_path=gate_run_path,
                        train_iters=train_iters,
                        evaluate_every_n_steps=evaluate_every_n_steps,
                        mixed_precision_mode=model_config.mixed_precision_mode,
                    )
                    command_dict[exp_name] = command
    return command_dict


def parse_commands_input(input_data: str) -> Dict[str, Any]:
    try:
        # Attempt to parse the input as JSON
        data = json.loads(input_data)

        if isinstance(data, dict):
            # If it's a dictionary, use it as-is

            return data
        elif isinstance(data, list):
            # If it's a list, generate a dictionary with auto-generated names
            return {f"exp-{i+1:03d}": cmd for i, cmd in enumerate(data)}
    except json.JSONDecodeError:
        # If JSON parsing fails, treat the input as a newline-separated list of commands
        print(
            f"Input data is not valid JSON. Attempting to parse as newline-separated list of commands."
        )
        return {
            f"exp-{i+1:03d}": cmd
            for i, cmd in enumerate(input_data.strip().split("\n"))
        }


def run_experiments(
    prefix: str = "debug",
    experiment_type: str = "all",
    accelerate_launch_path: str = "/opt/conda/envs/main/bin/accelerate-launch",
    gate_run_path: str = "/app/gate/run.py",
    # accelerate_launch_path: str = "accelerate launch",
    # gate_run_path: str = "gate/run.py",
    num_workers: int = 12,
    gpu_ids: Optional[Union[str, int]] = None,
    print_commands: bool = True,
    run_commands: bool = False,
    train_iters: int = 10000,
    evaluate_every_n_steps: int = 250,
    return_json: bool = False,
    seed_list: List[int] = [7],
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    selected_exp_name: Optional[List[str]] = None,
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
        return_json (bool): Whether to return the experiment commands as a JSON string instead of a list of command strings.

    Returns:
        experiment_dict (dict): A dictionary containing the experiment names as keys and the corresponding experiment commands as values.
    """

    if not sys.stdin.isatty():
        # If data is being piped to this script, read stdin
        selected_exp_name = parse_commands_input(sys.stdin.read())

    experiment_dict = {}

    experiment_configs: Dict[str, Any] = {
        "image-class": image_classification_config(),
        "few-shot": few_shot_learning_config(),
        "med-class": medical_image_classification_config(),
        "image-seg": image_segmentation_config(),
        "image-text": image_text_zero_shot_classification_config(),
        # "acdc": acdc_config,
        # "md": md_config,
        "rr": rr_config(),
        "rr-mm": rr_mm_config(),
        "video-class": video_classification_config(),
    }

    if experiment_type == "all":
        for experiment_type in experiment_configs:
            experiment_dict.update(
                generate_commands(
                    prefix=prefix,
                    seed_list=seed_list,
                    experiment_config=experiment_configs[experiment_type],
                    num_workers=num_workers,
                    accelerate_launch_path=accelerate_launch_path,
                    gate_run_path=gate_run_path,
                    gpu_ids=gpu_ids,
                    train_iters=train_iters,
                    evaluate_every_n_steps=evaluate_every_n_steps,
                )
            )
    elif "+" in experiment_type:
        experiment_types = experiment_type.split("+")
        for experiment_type in experiment_types:
            if experiment_type in experiment_configs:
                experiment_dict.update(
                    generate_commands(
                        prefix=prefix,
                        seed_list=seed_list,
                        experiment_config=experiment_configs[experiment_type],
                        num_workers=num_workers,
                        accelerate_launch_path=accelerate_launch_path,
                        gate_run_path=gate_run_path,
                        gpu_ids=gpu_ids,
                        train_iters=train_iters,
                        evaluate_every_n_steps=evaluate_every_n_steps,
                    )
                )
            else:
                logger.error("Invalid experiment type selected.")
                return
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
                train_iters=train_iters,
                evaluate_every_n_steps=evaluate_every_n_steps,
            )
        else:
            logger.error("Invalid experiment type selected.")
            return

    if print_commands:
        for experiment_name, experiment_command in experiment_dict.items():
            logger.info(f"{experiment_command}")

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
                    logger.info(line.decode().strip())

                for line in iter(process.stderr.readline, b""):
                    logger.info(line.decode().strip())

                # Wait for the process to complete and get the exit code
                process.communicate()
                exit_code = process.returncode

                if exit_code != 0:
                    logger.error(
                        f"Error executing {experiment_name}. Continuing with the next command."
                    )

    if start_idx is None:
        start_idx = 0

    if end_idx is None:
        end_idx = len(experiment_dict)

    if selected_exp_name is not None:
        selected_exp_name = [exp.lower() for exp in selected_exp_name.keys()]

        experiment_dict = {
            k: v
            for k, v in experiment_dict.items()
            if k.lower() in selected_exp_name
        }
        end_idx = len(experiment_dict)

    experiment_dict = dict(list(experiment_dict.items())[start_idx:end_idx])

    if return_json:
        return json.dumps(experiment_dict)

    return list(experiment_dict.values())


# Use Google Fire for command-line argument parsing
if __name__ == "__main__":
    fire.Fire(run_experiments)
