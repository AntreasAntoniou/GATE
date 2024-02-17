import os

import torch
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gate.data.image.segmentation.coco_10k import CLASSES, build_gate_dataset
from gate.metrics.segmentation import IoUMetric


def main():
    dataset_dict = build_gate_dataset(data_dir=os.environ.get("DATASET_DIR"))

    dataloader = DataLoader(
        dataset_dict["val"], batch_size=32, shuffle=False, num_workers=32
    )

    label_set = sorted(list(set(list(CLASSES.values()))))

    iou_metric = IoUMetric(
        num_classes=len(label_set),
        ignore_index=0,
        class_idx_to_name={idx: item for idx, item in enumerate(label_set)},
    )

    label_set = set()
    for item in tqdm(dataloader):
        image, labels = item["image"], item["labels"]

        # preds are labels in one hot format
        preds = labels.clone()
        labels = labels.squeeze()

        label_set.update(set(labels.unique().tolist()))
        print(label_set)
        print(len(label_set))
        iou_metric.update(preds, labels)

    metrics = iou_metric.compute_metrics()
    iou_metric.pretty_print(metrics=metrics)
    iou_metric.reset()  # Resetting the metrics after computation
    metrics_with_ignore = {
        k: v for k, v in metrics.items() if "per_class" not in k
    }


if __name__ == "__main__":
    main()
