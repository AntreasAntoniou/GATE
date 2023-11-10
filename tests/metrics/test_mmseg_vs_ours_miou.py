import numpy as np
import pytest
import torch
from mmseg.evaluation.metrics import IoUMetric as mmsegIoUMetric

from gate.metrics.segmentation import IoUMetric

# Creating dummy data for testing
num_classes = 255
ignore_index = 0
preds = torch.randint(1, num_classes, (8, 512, 512))
labels = torch.randint(1, num_classes, (8, 512, 512))
labels[:, :100, :100] = ignore_index  # Adding some ignore_index for testing


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


@pytest.mark.parametrize(
    "preds, labels, ignore_index",
    [(preds, labels, ignore_index), (preds, labels, None)],
)
def test_iou_metrics(preds, labels, ignore_index):
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

    metrics = your_metric.compute_metrics()
    your_metric.pretty_print()
    your_iou = metrics["mIoU"]

    # two decimal places rounding of your_iou
    your_iou = round(your_iou, 2)

    print(f"metrics = {metrics}")

    assert np.isclose(
        mmseg_iou, your_iou, rtol=1e-2
    ), f"mmseg: {mmseg_iou}, yours: {your_iou}"
