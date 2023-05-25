from playgrounds.experiment_manager import ExperimentRunner


if __name__ == "__main__":
    commands = {}
    for i in range(100):
        commands[f"test_{i}"] = f"python {__file__} test_gpu --gpu_id {i % 2}"
    er = ExperimentRunner(num_gpus=2, log_dir="/logs")
    er.run_experiments(commands)
