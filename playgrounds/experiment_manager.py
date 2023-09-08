import logging
import os
import subprocess
from threading import Lock, Thread
from typing import Dict

import fire
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")


class GPULocker:
    def __init__(self, num_gpus: int):
        self._locks = [Lock() for _ in range(num_gpus)]

    def run(self, cmd: str, gpu_id: int, log_path: str) -> None:
        with self._locks[gpu_id]:
            with open(log_path, "w") as f:
                logger.info(f"Starting command: {cmd} on GPU: {gpu_id}")
                subprocess.run(cmd, shell=True, stdout=f, stderr=f)
                logger.info(f"Finished command: {cmd} on GPU: {gpu_id}")

    def distribute(self, experiments: Dict[str, str], log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)

        threads = []
        for i, (exp_name, exp_cmd) in enumerate(experiments.items()):
            gpu_id = i % len(self._locks)
            log_path = os.path.join(log_dir, f"{exp_name}.log")
            cmd = f"{exp_cmd} --gpu-ids={gpu_id}"
            thread = Thread(target=self.run, args=(cmd, gpu_id, log_path))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


def main(num_gpus: int, log_dir: str):
    locker = GPULocker(num_gpus)

    experiments = {
        f"exp_{i}": f"python -c 'import torch;print(torch.cuda.get_device_properties({i % num_gpus}));import time;time.sleep(1)'"
        for i in range(100)
    }
    locker.distribute(experiments, log_dir)


if __name__ == "__main__":
    fire.Fire(main)
