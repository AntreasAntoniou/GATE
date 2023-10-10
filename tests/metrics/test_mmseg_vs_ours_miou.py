from typing import Optional

import numpy as np
import pytest
import torch
from mmseg.evaluation.metrics import IoUMetric as mmsegIoUMetric

from gate.metrics.segmentation import IoUMetric

# Creating dummy data for testing
num_classes = 21
ignore_index = 255
preds = torch.randint(0, num_classes, (8, 512, 512))
labels = torch.randint(0, num_classes, (8, 512, 512))
labels[:, :100, :100] = ignore_index  # Adding some ignore_index for testing


# Example usage
# Initialize with random predictions and labels
pred = torch.randint(0, 20, (100, 100))
label = torch.randint(0, 20, (100, 100))

class_idx_to_name = {
    i: f"class_{i}" for i in range(20)
}  # Example class index to name mapping
iou_metric = IoUMetric(num_classes=20, class_idx_to_name=class_idx_to_name)
iou_metric.update(pred, label)  # Update with predictions and labels
metrics = iou_metric.get_metrics()  # Get the computed metrics with class names

print(metrics)


# Helper function to compute metrics using mmseg's IoUMetric
def compute_metrics_with_mmseg(preds, labels, num_classes, ignore_index):
    mmseg_metric = mmsegIoUMetric(
        num_classes=num_classes, ignore_index=ignore_index
    )
    mmseg_metric.dataset_meta = {"classes": list(range(num_classes))}

    for pred, label in zip(preds, labels):
        data_samples = [
            {
                "pred_sem_seg": {"data": pred},
                "gt_sem_seg": {"data": label},
            }
        ]
        mmseg_metric.process(data_batch=None, data_samples=data_samples)

    metrics = mmseg_metric.compute_metrics(mmseg_metric.results)

    return metrics["mIoU"]


@pytest.mark.parametrize("preds, labels", [(preds, labels)])
def test_iou_metrics(preds, labels):
    num_classes = 21
    ignore_index = 255

    mmseg_iou = compute_metrics_with_mmseg(
        preds, labels, num_classes, ignore_index
    )

    your_metric = IoUMetric(
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_idx_to_name={i: str(i) for i in range(num_classes)},
    )

    for pred, label in zip(preds, labels):
        your_metric.update(pred, label)

    metrics = your_metric.get_metrics()
    your_metric.pretty_print()
    your_iou = metrics["mIoU"]

    print(f"metrics = {metrics}")

    assert np.isclose(
        mmseg_iou, your_iou, rtol=1e-2
    ), f"mmseg: {mmseg_iou}, yours: {your_iou}"


# Creating dummy data for testing
if __name__ == "__main__":
    num_classes = 21
    ignore_index = 255

    # Creating multiple batches of dummy data
    preds = [torch.randint(0, num_classes, (1, 512, 512)) for _ in range(5)]
    labels = [torch.randint(0, num_classes, (1, 512, 512)) for _ in range(5)]

    for label in labels:
        label[
            0, :100, :100
        ] = ignore_index  # Adding some ignore_index for testing

    pytest.main([__file__])
