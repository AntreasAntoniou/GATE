from collections import defaultdict
from enum import unique
from os import name
from typing import Optional

import fire
import pandas as pd
import yaml
from rich import print

from gate import data

ensemble_series = 1

dataset_to_target_metrics = {
    "imagenet1k": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
    },
    "food101": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
    },
    "cifar100": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
    },
    "places365": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
    },
    "whale-dolphin": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_5_individual-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_individual-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_species-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_5_species-epoch-mean",
    },
    "fungi": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "aircraft": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "omniglot": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "cubirds": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "dtextures": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "vgg": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "mini": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "diabetic": {
        f"testing/ensemble_{ensemble_series}/auc-macro",
        f"testing/ensemble_{ensemble_series}/bs-macro",
        f"testing/ensemble_{ensemble_series}/aps-macro",
    },
    "ham10k": {
        f"testing/ensemble_{ensemble_series}/auc-macro",
        f"testing/ensemble_{ensemble_series}/bs-macro",
        f"testing/ensemble_{ensemble_series}/aps-macro",
    },
    "chexpert": {
        f"testing/ensemble_{ensemble_series}/auc-macro",
        f"testing/ensemble_{ensemble_series}/bs-macro",
        f"testing/ensemble_{ensemble_series}/aps-macro",
    },
    "ade20k": {
        f"testing/ensemble_{ensemble_series}/mIoU",
        f"testing/ensemble_{ensemble_series}/mean_accuracy",
        f"testing/ensemble_{ensemble_series}/overall_accuracy",
        f"testing/ensemble_{ensemble_series}/dice_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/focal_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/ce_loss-epoch-mean",
    },
    "cityscapes": {
        f"testing/ensemble_{ensemble_series}/mIoU",
        f"testing/ensemble_{ensemble_series}/mean_accuracy",
        f"testing/ensemble_{ensemble_series}/overall_accuracy",
        f"testing/ensemble_{ensemble_series}/dice_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/focal_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/ce_loss-epoch-mean",
    },
    "coco-10k": {
        f"testing/ensemble_{ensemble_series}/mIoU",
        f"testing/ensemble_{ensemble_series}/mean_accuracy",
        f"testing/ensemble_{ensemble_series}/overall_accuracy",
        f"testing/ensemble_{ensemble_series}/dice_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/focal_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/ce_loss-epoch-mean",
    },
    "coco-164k": {
        f"testing/ensemble_{ensemble_series}/mIoU",
        f"testing/ensemble_{ensemble_series}/mean_accuracy",
        f"testing/ensemble_{ensemble_series}/overall_accuracy",
        f"testing/ensemble_{ensemble_series}/dice_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/focal_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/ce_loss-epoch-mean",
    },
    "pascal": {
        f"testing/ensemble_{ensemble_series}/mIoU",
        f"testing/ensemble_{ensemble_series}/mean_accuracy",
        f"testing/ensemble_{ensemble_series}/overall_accuracy",
        f"testing/ensemble_{ensemble_series}/dice_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/focal_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/ce_loss-epoch-mean",
    },
    "nyu": {
        f"testing/ensemble_{ensemble_series}/mIoU",
        f"testing/ensemble_{ensemble_series}/mean_accuracy",
        f"testing/ensemble_{ensemble_series}/overall_accuracy",
        f"testing/ensemble_{ensemble_series}/dice_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/focal_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/ce_loss-epoch-mean",
    },
    "winoground": {
        f"testing/ensemble_{ensemble_series}/text_to_image_accuracy-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_accuracy-epoch-mean",
        f"testing/ensemble_{ensemble_series}/text_to_image_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_loss-epoch-mean",
    },
    "flickr30k": {
        f"testing/ensemble_{ensemble_series}/text_to_image_accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/text_to_image_accuracy-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_accuracy-epoch-mean",
        f"testing/ensemble_{ensemble_series}/text_to_image_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_loss-epoch-mean",
    },
    "newyorkercaptioncontest": {
        f"testing/ensemble_{ensemble_series}/text_to_image_accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/text_to_image_accuracy-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_accuracy-epoch-mean",
    },
    "pokemonblipcaptions": {
        f"testing/ensemble_{ensemble_series}/text_to_image_accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/text_to_image_accuracy-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_accuracy-epoch-mean",
        f"testing/ensemble_{ensemble_series}/text_to_image_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/image_to_text_loss-epoch-mean",
    },
    "hmdb51": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "kinetics": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "ucf": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "iwildcam": {
        f"testing/ensemble_{ensemble_series}/mae_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/mse_loss-epoch-mean",
    },
    "clevr": {
        f"testing/ensemble_{ensemble_series}/loss_yes_no-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss_count-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss_colour-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss_shape-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss_count-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss_size-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss_material-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_yes_no-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_count-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_colour-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_shape-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_size-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_1_material-epoch-mean",
        # f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        # f"testing/ensemble_{ensemble_series}/accuracy_top_5_yes_no-epoch-mean",
        # f"testing/ensemble_{ensemble_series}/accuracy_top_5_count-epoch-mean",
        # f"testing/ensemble_{ensemble_series}/accuracy_top_5_colour-epoch-mean",
        # f"testing/ensemble_{ensemble_series}/accuracy_top_5_shape-epoch-mean",
        # f"testing/ensemble_{ensemble_series}/accuracy_top_5_size-epoch-mean",
        # f"testing/ensemble_{ensemble_series}/accuracy_top_5_material-epoch-mean",
    },
    "clevr-math": {
        f"testing/ensemble_{ensemble_series}/accuracy_top_1-epoch-mean",
        f"testing/ensemble_{ensemble_series}/accuracy_top_5-epoch-mean",
        f"testing/ensemble_{ensemble_series}/loss-epoch-mean",
    },
    "acdc": {
        f"testing/ensemble_{ensemble_series}-/dice_loss-epoch-mean",
        f"testing/ensemble_{ensemble_series}/mIoU",
        f"testing/ensemble_{ensemble_series}/mean_accuracy",
        f"testing/ensemble_{ensemble_series}/overall_accuracy",
    },
}


def match_model(exp_model_name: str) -> Optional[str]:
    model_names = [
        "AR-ViT-B16-224",
        "BART",
        "BERT",
        "CLIP-B16-224",
        # "CLIP-B16-224HF-image",
        "CLIP-B16-224HF-text",
        "ConvNextV2-Base",
        "DINO-B16-224",
        "DeiT3-B16-224",
        "EffFormer-s0",
        "EffV2-RW-S",
        "Flex-B-1200EP",
        # "IJEPA-Huge-P14-224",
        "Laion-B16-224",
        "MPNET",
        "RNX50-32x4A1",
        "SIGLIP-P16-224",
        "SViT-B16-224",
        # "W2V2",
        "Whisper",
    ]

    found = False
    for m in model_names:
        if m in exp_model_name:
            exp_model_name = m
            found = True
            break
    if not found:
        print(f"not found {exp_model_name}")
        return None
    return exp_model_name


def extract_model_name(exp_name: str, dataset_name: str) -> str:
    model_name = exp_name.replace(dataset_name, "")
    model_name = "-".join(model_name.split("-")[1:-1])
    model_name = model_name.replace("--", "").replace("classification-", "")
    model_name = model_name.replace("-fs-", "").replace("happy", "")
    model_name = model_name.replace("retionopathy", "")
    model_name = model_name[1:] if model_name[0] == "-" else model_name
    model_name = model_name[1:] if model_name[0] == "-" else model_name
    model_name = match_model(model_name)
    return model_name


def check_validity(dataset_to_metrics: dict, dataset_to_target_metrics: dict):
    for dataset in dataset_to_target_metrics:
        for metric in dataset_to_target_metrics[dataset]:
            if metric not in dataset_to_metrics[dataset]:
                print(f"{dataset} {metric} not in source metrics")
                print(f"Instead we have {sorted(dataset_to_metrics[dataset])}")


def get_non_empty_columns_and_values(df, index):
    non_empty_columns = df.loc[index].notnull()
    non_empty_columns = non_empty_columns[non_empty_columns == True]
    non_empty_columns = non_empty_columns.index.tolist()
    non_empty_values = df.loc[index, non_empty_columns].tolist()
    return dict(zip(non_empty_columns, non_empty_values))


def get_dataset_name(exp_dataset_name: str, exp_name) -> str:
    exp_dataset_name = (
        exp_dataset_name if exp_dataset_name != "happy" else "whale-dolphin"
    )
    exp_dataset_name = (
        "clevr-math"
        if "clevr-math" in exp_name
        else "clevr" if exp_dataset_name == "clevr" else exp_dataset_name
    )
    exp_dataset_name = (
        "coco-10k"
        if "coco-10k" in exp_name
        else "coco-164k" if exp_dataset_name == "coco" else exp_dataset_name
    )
    return exp_dataset_name


def load_dataframe(source_csv_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(source_csv_filepath)
    exp_series = df.iloc[:, 0].tolist()
    exp_names = df.iloc[:, 2].tolist()
    dataset_names = df.iloc[:, 1].tolist()
    metrics = df.columns[4:].tolist()

    dataset_to_metrics = defaultdict(set)

    for idx, row in df.iterrows():
        exp_dataset_name = get_dataset_name(row[1], row[2])
        dataset_to_metrics[exp_dataset_name].update(
            get_non_empty_columns_and_values(df, idx).keys()
        )

    check_validity(dataset_to_metrics, dataset_to_target_metrics)

    model_to_series_dataset_to_metrics = defaultdict(dict)
    unique_dataset_names = set()
    for idx, row in df.iterrows():
        results_dict = get_non_empty_columns_and_values(df, idx)
        exp_name = results_dict["Experiment-name"]
        exp_dataset_name = results_dict["Dataset-name"]
        exp_series = results_dict["Experiment-series"]
        # exp_dataset_name = (
        #     exp_dataset_name if exp_dataset_name != "happy" else "whale-dolphin"
        # )
        exp_dataset_name = get_dataset_name(exp_dataset_name, exp_name)
        model_name = extract_model_name(
            exp_name=exp_name, dataset_name=exp_dataset_name
        )
        seed = (
            exp_name.split("-")[-1]
            if "hpo" not in exp_name
            else exp_name.split("-")[-3:]
        )
        exp_series = f"{exp_series}-{seed}"
        if exp_dataset_name == "medical":
            continue
        unique_dataset_names.add(exp_dataset_name)

        for metric in dataset_to_metrics[exp_dataset_name]:
            if (
                exp_dataset_name in dataset_to_target_metrics
                and metric in dataset_to_target_metrics[exp_dataset_name]
            ):
                if (
                    exp_series
                    not in model_to_series_dataset_to_metrics[model_name]
                ):
                    model_to_series_dataset_to_metrics[model_name][
                        exp_series
                    ] = dict()
                model_to_series_dataset_to_metrics[model_name][exp_series][
                    exp_dataset_name
                ] = model_to_series_dataset_to_metrics[model_name][
                    exp_series
                ].get(
                    exp_dataset_name, dict()
                )
                if (
                    metric
                    not in model_to_series_dataset_to_metrics[model_name][
                        exp_series
                    ][exp_dataset_name]
                ):
                    model_to_series_dataset_to_metrics[model_name][exp_series][
                        exp_dataset_name
                    ][metric] = dict(
                        value=results_dict.get(metric, None), name=exp_name
                    )
    print(model_to_series_dataset_to_metrics)
    unique_dataset_names = sorted(unique_dataset_names)
    clean_model_to_series_dataset_to_metrics = defaultdict(dict)

    for model in model_to_series_dataset_to_metrics:
        for series in model_to_series_dataset_to_metrics[model]:
            for dataset in unique_dataset_names:
                if (
                    dataset
                    not in model_to_series_dataset_to_metrics[model][series]
                    and dataset
                    not in clean_model_to_series_dataset_to_metrics[model]
                ):
                    clean_model_to_series_dataset_to_metrics[model][
                        dataset
                    ] = dict()
                    continue

                if (
                    dataset
                    not in model_to_series_dataset_to_metrics[model][series]
                ):
                    continue

                if (
                    dataset
                    not in clean_model_to_series_dataset_to_metrics[model]
                ):
                    clean_model_to_series_dataset_to_metrics[model][
                        dataset
                    ] = model_to_series_dataset_to_metrics[model][series][
                        dataset
                    ]
                else:
                    for metric in model_to_series_dataset_to_metrics[model][
                        series
                    ][dataset]:
                        if (
                            metric
                            not in clean_model_to_series_dataset_to_metrics[
                                model
                            ][dataset]
                        ):
                            clean_model_to_series_dataset_to_metrics[model][
                                dataset
                            ][metric] = model_to_series_dataset_to_metrics[
                                model
                            ][
                                series
                            ][
                                dataset
                            ][
                                metric
                            ]
                        else:
                            existing_value = (
                                clean_model_to_series_dataset_to_metrics[
                                    model
                                ][dataset][metric]["value"]
                            )
                            incoming_value = (
                                model_to_series_dataset_to_metrics[model][
                                    series
                                ][dataset][metric]["value"]
                            )

                            if existing_value is None:
                                clean_model_to_series_dataset_to_metrics[
                                    model
                                ][dataset][
                                    metric
                                ] = model_to_series_dataset_to_metrics[
                                    model
                                ][
                                    series
                                ][
                                    dataset
                                ][
                                    metric
                                ]
                            elif incoming_value is None:
                                continue

                            elif "loss" in metric:
                                if incoming_value < existing_value:
                                    clean_model_to_series_dataset_to_metrics[
                                        model
                                    ][dataset][
                                        metric
                                    ] = model_to_series_dataset_to_metrics[
                                        model
                                    ][
                                        series
                                    ][
                                        dataset
                                    ][
                                        metric
                                    ]
                            elif "accuracy" in metric:

                                if incoming_value > existing_value:
                                    clean_model_to_series_dataset_to_metrics[
                                        model
                                    ][dataset][
                                        metric
                                    ] = model_to_series_dataset_to_metrics[
                                        model
                                    ][
                                        series
                                    ][
                                        dataset
                                    ][
                                        metric
                                    ]
                            elif "macro" in metric:
                                if incoming_value > existing_value:
                                    clean_model_to_series_dataset_to_metrics[
                                        model
                                    ][dataset][
                                        metric
                                    ] = model_to_series_dataset_to_metrics[
                                        model
                                    ][
                                        series
                                    ][
                                        dataset
                                    ][
                                        metric
                                    ]
                            elif "miou" in metric.lower():
                                if incoming_value > existing_value:
                                    clean_model_to_series_dataset_to_metrics[
                                        model
                                    ][dataset][
                                        metric
                                    ] = model_to_series_dataset_to_metrics[
                                        model
                                    ][
                                        series
                                    ][
                                        dataset
                                    ][
                                        metric
                                    ]
    print(clean_model_to_series_dataset_to_metrics)


def main(source_csv_filepath: str):
    load_dataframe(source_csv_filepath)


if __name__ == "__main__":
    fire.Fire(main)
