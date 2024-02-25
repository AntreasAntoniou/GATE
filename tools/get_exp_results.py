import collections
import csv
import os
import pathlib
import sys
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import Opt
from rich import print
from tqdm.auto import tqdm

import wandb

plt.style.use("ggplot")


def convert_to_datetime(date_string):
    """Convert the string to a datetime object"""
    date_object = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
    return date_object


def get_runs(api, project_name):
    """Get runs from a specific project"""
    runs = api.runs(project_name)
    return runs


def get_experiment_data(runs):
    """Get experiment data from runs"""
    exp_name_to_time_dict = {}
    exp_name_to_data_dict = {}
    all_keys = set()
    for run in tqdm(runs):
        metric_dict = run.summary._json_dict
        timestamp = convert_to_datetime(run.heartbeatAt)
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        if "exp_name" in config:
            exp_name = config["exp_name"]
            if exp_name in exp_name_to_time_dict:
                if (
                    timestamp > exp_name_to_time_dict[exp_name]
                    and "testing" in metric_dict
                ):
                    exp_name_to_time_dict[exp_name] = timestamp
                    exp_name_to_data_dict[exp_name] = [metric_dict, config]
                    for key in metric_dict.keys():
                        if "testing" in key:
                            all_keys.add(key)
                else:
                    continue

            exp_name_to_data_dict[exp_name] = [metric_dict, config]
            for key in metric_dict.keys():
                if "testing" in key:
                    all_keys.add(key)
    return exp_name_to_data_dict, all_keys


def filter_keys(all_keys, include_keys=None, exclude_keys=None):
    """Filter keys based on certain conditions"""
    selected_keys = set()
    for key in sorted(all_keys):
        if (
            # "shape" not in key
            # and "colour" not in key
            "logits" not in key
            # and "count" not in key
            # and "material" not in key
            # and "yes_no" not in key
            # and "size" not in key
            and "similarities" not in key
        ):
            selected_keys.add(key)
    if include_keys:
        selected_keys = [
            item
            for item in selected_keys
            if any(str(key) in item for key in include_keys)
        ]

    if exclude_keys:
        selected_keys = [
            item
            for item in selected_keys
            if all(str(key) not in item for key in exclude_keys)
        ]

    return selected_keys


def get_experiment_dict(name_list, config_list, summary_list, selected_keys):
    """Get experiment dictionary"""
    from collections import defaultdict

    exp_dict = defaultdict(dict)
    for name, config, metric_dict in zip(name_list, config_list, summary_list):
        if any([key in metric_dict.keys() for key in selected_keys]):
            for key in selected_keys:
                if key in metric_dict:
                    exp_dict[name][key.replace("-/", "/")] = metric_dict[key]
    return exp_dict


def aggregate_experiments(
    experiments: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate experiments"""
    aggregated = collections.defaultdict(lambda: collections.defaultdict(list))
    for experiment_name, metrics in experiments.items():
        base_name = experiment_name
        for metric, value in metrics.items():
            aggregated[base_name][metric].append(value)
    return aggregated


def create_csv(
    output_filename: str,
    aggregated_experiments: Dict[str, Dict[str, List[float]]],
) -> None:
    """Create CSV from aggregated experiments"""
    unique_metrics = set()
    for _, metrics in aggregated_experiments.items():
        unique_metrics.update(metrics.keys())
    rows = []
    for experiment_name, metrics in aggregated_experiments.items():
        experiment_parts = experiment_name.split("-", 2)
        experiment_series, dataset_name = experiment_parts[:2]
        row = {
            "Experiment-series": experiment_series,
            "Dataset-name": dataset_name,
            "Experiment-name": experiment_name,
            "count": 0,
        }
        for metric in unique_metrics:
            values = metrics.get(metric, [])
            count = len(values)

            if count == 0:
                continue

            if isinstance(values[0], dict):
                continue

            if "NaN" in values:
                continue
            mean_value = sum(values) / count if count > 0 else None
            row[metric] = mean_value
            row["count"] = max(row["count"], count)
        rows.append(row)
    with open(output_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Experiment-series",
            "Dataset-name",
            "Experiment-name",
            "count",
        ] + sorted(unique_metrics)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


OptListStr = Optional[List[str]]


def main(
    project_name: str,
    output_filename: str = "experiments_summary.csv",
    include_keys: OptListStr = None,
    exclude_keys: OptListStr = None,
    exp_name_include: OptListStr = None,
    exp_name_exclude: OptListStr = None,
):
    """Main function"""
    api = wandb.Api()
    runs = get_runs(api, project_name)
    exp_name_to_data_dict, all_keys = get_experiment_data(runs)
    selected_keys = filter_keys(
        all_keys, include_keys=include_keys, exclude_keys=exclude_keys
    )
    exp_dict = get_experiment_dict(
        exp_name_to_data_dict.keys(),
        [item[1] for item in exp_name_to_data_dict.values()],
        [item[0] for item in exp_name_to_data_dict.values()],
        selected_keys,
    )
    if exp_name_include:
        exp_dict = {
            key: value
            for key, value in exp_dict.items()
            if any(str(item) in key for item in exp_name_include)
        }
    if exp_name_exclude:
        exp_dict = {
            key: value
            for key, value in exp_dict.items()
            if all(str(item) not in key for item in exp_name_exclude)
        }
    aggregated_experiments = aggregate_experiments(exp_dict)
    create_csv(output_filename, aggregated_experiments)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
