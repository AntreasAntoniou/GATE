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
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    utilization = nvmlDeviceGetUtilizationRates(handle)

    return (
        memory_info.used / memory_info.total
    ) * 100 <= memory_threshold and utilization.gpu <= util_threshold


def get_gpu_processes(memory_threshold=5, util_threshold=10):
    nvmlInit()
    gpu_processes = {}
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        if is_gpu_available(handle, memory_threshold, util_threshold):
            gpu_processes[
                str(i)
            ] = None  # No process is currently using this GPU
    nvmlShutdown()
    return gpu_processes


def run_command_on_gpu(command, gpu_id, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    stdout_file = open(f"{os.environ['LOG_DIR']}/{exp_name}.stdout", "w")
    stderr_file = open(f"{os.environ['LOG_DIR']}/{exp_name}.stderr", "w")
    return subprocess.Popen(
        command, shell=True, stdout=stdout_file, stderr=stderr_file
    )  # Return the process handle


def run_commands(command_dict: Dict, memory_threshold=5, util_threshold=10):
    gpu_processes = get_gpu_processes(memory_threshold, util_threshold)

    with tqdm(total=len(command_dict), smoothing=0.0) as pbar:
        while command_dict:
            for gpu_id, process in gpu_processes.items():
                # If the GPU is available or the process using it has completed
                if process is None or process.poll() is not None:
                    if command_dict:
                        command_name, command = command_dict.popitem()
                        pbar.set_description(
                            f"Running command on GPU {gpu_id}: {command}"
                        )
                        gpu_processes[gpu_id] = run_command_on_gpu(
                            command=command,
                            gpu_id=gpu_id,
                            exp_name=command_name,
                        )
                        pbar.update(1)
                    else:
                        break  # No more commands to run

            # Sleep for a bit before checking GPU availability again
            pbar.set_description("Waiting for GPUs to become available... 🎣")
            time.sleep(1)

    # Wait for all processes to complete
    for gpu_id, process in gpu_processes.items():
        if process is not None:
            process.wait()


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
        return {
            f"exp-{i+1:03d}": cmd
            for i, cmd in enumerate(input_data.strip().split("\n"))
        }


def main(
    memory_threshold: int = 5,
    util_threshold: int = 10,
    log_dir: Optional[Union[str, pathlib.Path]] = pathlib.Path("logs/"),
):
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

    # save the commands in a txt file
    with open(f"{os.environ['LOG_DIR']}/commands.txt", "w") as f:
        for command_name, command in command_dict.items():
            f.write(f"{command_name}: {command}\n")

    # Now, you can pass the list of commands and other arguments to your function
    run_commands(
        command_dict,
        memory_threshold=memory_threshold,
        util_threshold=util_threshold,
    )


if __name__ == "__main__":
    fire.Fire(main)
