import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Union

import fire
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)
from tqdm.auto import tqdm


def is_gpu_available(handle, memory_threshold=5, util_threshold=10):
    """
    Check if the GPU is available based on memory and utilization thresholds.

    Args:
        handle: The handle of the GPU to check.
        memory_threshold: The maximum memory usage percentage that the GPU can have to be considered available.
        util_threshold: The maximum GPU utilization percentage that the GPU can have to be considered available.

    Returns:
        True if the GPU is available, False otherwise.
    """
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    utilization = nvmlDeviceGetUtilizationRates(handle)

    return (
        memory_info.used / memory_info.total
    ) * 100 <= memory_threshold and utilization.gpu <= util_threshold


def get_gpu_processes(memory_threshold=5, util_threshold=10):
    """
    Get the IDs of all available GPUs.

    Args:
        memory_threshold: The maximum memory usage percentage that a GPU can have to be considered available.
        util_threshold: The maximum GPU utilization percentage that a GPU can have to be considered available.

    Returns:
        A list of the IDs of all available GPUs.
    """
    nvmlInit()
    available_gpus = [
        str(i)
        for i in range(nvmlDeviceGetCount())
        if is_gpu_available(
            nvmlDeviceGetHandleByIndex(i), memory_threshold, util_threshold
        )
    ]
    nvmlShutdown()
    return available_gpus


def run_command_on_gpu(command, gpu_ids, exp_name):
    """
    Run a command on the specified GPUs.

    Args:
        command: The command to run.
        gpu_ids: A list of the IDs of the GPUs to run the command on.
        exp_name: The name of the experiment.

    Returns:
        The handle of the process running the command.
    """
    command = command.replace(
        f"accelerate launch",
        "accelerate launch --gpu_ids=" + ",".join(gpu_ids),
    )
    command = command.replace("29012024", "30012024")
    stdout_file = open(f"{os.environ['LOG_DIR']}/{exp_name}.stdout.log", "w")
    stderr_file = open(f"{os.environ['LOG_DIR']}/{exp_name}.stderr.log", "w")

    return subprocess.Popen(
        command, shell=True, stdout=stdout_file, stderr=stderr_file
    )


def run_commands(
    command_dict: Dict, num_gpus: int, memory_threshold=5, util_threshold=10
):
    """
    Run multiple commands on available GPUs.

    Args:
        command_dict: A dictionary mapping command names to commands.
        num_gpus: The number of GPUs to use for each experiment.
        memory_threshold: The maximum memory usage percentage that a GPU can have to be considered available.
        util_threshold: The maximum GPU utilization percentage that a GPU can have to be considered available.
    """
    available_gpus = get_gpu_processes(memory_threshold, util_threshold)
    with tqdm(total=len(command_dict), smoothing=0.0) as pbar:
        while command_dict:
            if len(available_gpus) >= num_gpus:
                gpu_ids = available_gpus[:num_gpus]

                command_name, command = command_dict.popitem()

                print(f"Running command {command_name} on GPUs {gpu_ids}")
                _ = run_command_on_gpu(
                    command=command,
                    gpu_ids=gpu_ids,
                    exp_name=command_name,
                )
                available_gpus = [
                    gpu for gpu in available_gpus if gpu not in gpu_ids
                ]
                pbar.update(1)
                pbar.set_description(f"Running on GPUs {gpu_ids}")

            else:
                time.sleep(240)
                available_gpus = get_gpu_processes(
                    memory_threshold, util_threshold
                )

            # Sleep for a bit before assigning more GPUs
            pbar.set_description("Waiting for GPUs to become available... 🎣")


def parse_commands_input(input_data: str) -> Dict[str, Any]:
    """
    Parse the commands input.

    If the input data is JSON, it is parsed as such. If it is a string, it is split by newline characters.

    Args:
        input_data: The input data containing the commands.

    Returns:
        A dictionary mapping command names to commands.
    """
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
        return {
            f"exp-{i+1:03d}": cmd
            for i, cmd in enumerate(input_data.strip().split("\n"))
        }


def main(
    num_gpus: int,
    memory_threshold: int = 5,
    util_threshold: int = 10,
    log_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """
    The main function of the program.

    This function takes in the number of GPUs to use, memory and utilization thresholds, and a log directory.
    It parses the commands from the input, runs them on the GPUs, and logs the results.

    Args:
        num_gpus: The number of GPUs to use for the experiments.
        memory_threshold: The maximum memory usage percentage that a GPU can have to be considered available.
        util_threshold: The maximum GPU utilization percentage that a GPU can have to be considered available.
        log_dir: The directory to log the results to.
    """
    if log_dir is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"logs/{current_time}/"

    if not isinstance(log_dir, pathlib.Path):
        log_dir = pathlib.Path(log_dir)

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    os.environ["LOG_DIR"] = (
        log_dir.as_posix() if isinstance(log_dir, pathlib.Path) else log_dir
    )
    command_dict = {}
    if not sys.stdin.isatty():
        # If data is being piped to this script, read stdin
        command_dict = parse_commands_input(sys.stdin.read())

    with open(f"{os.environ['LOG_DIR']}/commands.txt", "w") as f:
        for command_name, command in command_dict.items():
            f.write(f"{command_name}: {command}\n")

    # Now, you can pass the list of commands and other arguments to your function
    run_commands(
        command_dict,
        num_gpus,
        memory_threshold=memory_threshold,
        util_threshold=util_threshold,
    )


if __name__ == "__main__":
    fire.Fire(main)
