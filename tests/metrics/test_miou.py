import numpy as np
import torch
from mmseg.evaluation.metrics import IoUMetric


def test_IoUMetric():
    # Create an instance of your class
    metric = IoUMetric(ignore_index=255, iou_metrics=["mIoU"], prefix="test")

    # Create some dummy data
    data_batch = {}

    metric.dataset_meta = {"classes": [0, 1]}

    data_samples = [
        {
            "pred_sem_seg": {"data": torch.tensor([[[0, 1], [1, 0]]])},
            "gt_sem_seg": {"data": torch.tensor([[[0, 1], [1, 0]]])},
            "img_path": "dummy_path",
        }
    ]

    # Call the process method
    metric.process(data_batch, data_samples)

    print(metric.results)

    # Call the compute_metrics method
    metrics = metric.compute_metrics(metric.results)

    print(metrics)

    # Check the metrics
    assert np.allclose(metrics["aAcc"], 100)
    assert np.allclose(metrics["mIoU"], 100)
