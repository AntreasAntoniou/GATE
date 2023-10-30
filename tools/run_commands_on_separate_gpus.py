import os
import subprocess
import sys
import time

import fire
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


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


def run_command_on_gpu(command, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return subprocess.Popen(command, shell=True)  # Return the process handle


def run_commands(commands, memory_threshold=5, util_threshold=10):
    gpu_processes = get_gpu_processes(memory_threshold, util_threshold)

    while commands:
        for gpu_id, process in gpu_processes.items():
            # If the GPU is available or the process using it has completed
            if process is None or process.poll() is not None:
                if commands:
                    command = commands.pop(0)
                    print(f"Running command on GPU {gpu_id}: {command}")
                    gpu_processes[gpu_id] = run_command_on_gpu(command, gpu_id)
                else:
                    break  # No more commands to run

        # Sleep for a bit before checking GPU availability again
        print("Waiting for GPUs to become available...")
        time.sleep(1)

    # Wait for all processes to complete
    for gpu_id, process in gpu_processes.items():
        if process is not None:
            process.wait()


def main(memory_threshold=5, util_threshold=10):
    commands = []
    if not sys.stdin.isatty():
        # If data is being piped to this script, read stdin
        commands.extend(line.strip() for line in sys.stdin)

    # Now, you can pass the list of commands and other arguments to your function
    run_commands(
        commands,
        memory_threshold=memory_threshold,
        util_threshold=util_threshold,
    )


if __name__ == "__main__":
    fire.Fire(main)
