import os

import matplotlib.pyplot as plt
import torch
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gate.data.image.segmentation.cityscapes import CLASSES as CLASSES_HUG
from gate.data.image.segmentation.cityscapes import (
    build_gate_dataset as build_gate_dataset_hug,
)
from gate.data.image.segmentation.cityscapes_pytorch import (
    CLASSES,
    build_gate_dataset,
)
from gate.metrics.segmentation import IoUMetric


def main():
    dataset_dict = build_gate_dataset(data_dir=os.environ.get("DATASET_DIR"))
    dataset_hug_dict = build_gate_dataset_hug(
        data_dir=os.environ.get("DATASET_DIR")
    )

    dataloader = DataLoader(
        dataset_dict["val"], batch_size=32, shuffle=False, num_workers=32
    )
    dataloader_hug = DataLoader(
        dataset_hug_dict["val"], batch_size=32, shuffle=False, num_workers=32
    )

    label_set = sorted(list(set([item.id for item in CLASSES])))

    iou_metric = IoUMetric(
        num_classes=len(label_set),
        ignore_index=0,
        class_idx_to_name={idx: item for idx, item in enumerate(label_set)},
    )

    label_set = set()
    idx = 0
    label_frequency_dict = {}
    label_frequency_dict_hug = {}
    dataset_sizes = {
        "original_val": len(dataset_dict["val"]),
        "hug_val": len(dataset_hug_dict["val"]),
    }

    for item, item_hug in tqdm(zip(dataloader, dataloader_hug)):
        image, labels = item["image"], item["labels"]
        image_hug, labels_hug = item_hug["image"], item_hug["labels"]

        # diff_image = torch.abs(image - image_hug)
        # diff_labels = torch.abs(labels - labels_hug)

        # visualize images and diff
        # canvas = torch.cat([image, image_hug, diff_image], dim=2)
        # canvas = canvas.permute(1, 2, 0).numpy()
        # canvas = (canvas * 255).astype("uint8")

        # save the canvas
        # plt.imsave(f"canvas_{idx}.png", canvas)
        idx += 1
        # print(f"Image: {diff_image.max()}, Labels: {diff_labels.max()}")

        # preds are labels in one hot format
        preds = labels.clone()
        labels = labels.squeeze()
        labels_hug = labels_hug.squeeze()

        label_set.update(set(labels.unique().tolist()))
        print(label_set)
        print(len(label_set))
        label_freq = torch.bincount(labels.view(-1))
        # get keys and frequency of each label
        label_keys = torch.nonzero(label_freq).view(-1)
        label_values = label_freq[label_keys]

        label_freq_hug = torch.bincount(labels_hug.view(-1))
        # get keys and frequency of each label
        label_keys_hug = torch.nonzero(label_freq_hug).view(-1)
        label_values_hug = label_freq_hug[label_keys_hug]

        label_frequency_dict = {
            label: label_frequency_dict.get(label, 0) + 1
            for label in labels.view(-1).tolist()
        }
        label_frequency_dict_hug = {
            label: label_frequency_dict_hug.get(label, 0) + 1
            for label in labels_hug.view(-1).tolist()
        }
        print(
            f"Label Frequency Dict: {label_frequency_dict}, Hug: {label_frequency_dict_hug}"
        )
        iou_metric.update(preds, labels)

    metrics = iou_metric.compute_metrics()
    iou_metric.pretty_print(metrics=metrics)
    iou_metric.reset()  # Resetting the metrics after computation
    metrics_with_ignore = {
        k: v for k, v in metrics.items() if "per_class" not in k
    }


if __name__ == "__main__":
    main()
