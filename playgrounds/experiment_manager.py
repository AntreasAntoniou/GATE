import fire
import os
import queue
import subprocess
import threading
import time
import datetime
import torch
from rich.console import Console
from tqdm import tqdm


class ExperimentRunner:
    def __init__(self, num_gpus: int, log_dir: str):
        self.num_gpus = num_gpus
        self.log_dir = log_dir
        self.gpu_queue = queue.Queue()

        for i in range(num_gpus):
            self.gpu_queue.put(i)

    def run_experiment(self, exp_name, exp_command):
        gpu_id = (
            self.gpu_queue.get()
        )  # this will block until a GPU is available
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, f"{exp_name}.log"), "w") as f:
                # Add the --gpu-ids argument to the command
                complete_command = f"{exp_command} --gpu-ids={gpu_id}"
                subprocess.run(complete_command.split(), stdout=f, stderr=f)
        finally:
            self.gpu_queue.put(gpu_id)

    def run_experiments(self, experiments):
        with tqdm(total=len(experiments)) as pbar:
            threads = []
            for exp_name, exp_command in experiments.items():
                thread = threading.Thread(
                    target=self.run_experiment, args=(exp_name, exp_command)
                )
                thread.start()
                threads.append(thread)
                pbar.update(1)

            for thread in threads:
                thread.join()  # wait for all threads to finish

    def test_gpu(self, gpu_id: int):
        device = torch.device(f"cuda:{gpu_id}")
        properties = torch.cuda.get_device_properties(device)
        console = Console()
        console.print(
            f"GPU {gpu_id} memory: {properties.total_memory / 1024 ** 3:.2f} GB"
        )
        console.print(f"Current time: {datetime.datetime.now()}")


if __name__ == "__main__":
    fire.Fire(ExperimentRunner)
