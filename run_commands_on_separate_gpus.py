import os
import subprocess
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


def get_available_gpus(memory_threshold=5, util_threshold=10):
    nvmlInit()
    available_gpus = []
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        if is_gpu_available(handle, memory_threshold, util_threshold):
            available_gpus.append(str(i))
    nvmlShutdown()
    return available_gpus


def run_command_on_gpu(command, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    subprocess.run(command, shell=True)


def run_commands(commands, memory_threshold=5, util_threshold=10):
    while commands:
        available_gpus = get_available_gpus(memory_threshold, util_threshold)
        if not available_gpus:
            print("No available GPU. Retrying in 1 second...")
            time.sleep(1)
            continue

        # Run commands on available GPUs
        for gpu_id in available_gpus:
            if commands:
                command = commands.pop(0)
                print(f"Running command on GPU {gpu_id}: {command}")
                run_command_on_gpu(command, gpu_id)
            else:
                break


if __name__ == "__main__":
    fire.Fire(run_commands)
