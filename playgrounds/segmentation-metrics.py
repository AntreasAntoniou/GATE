from mmengine.evaluator import BaseMetric
from mmseg.evaluation.metrics import IoUMetric
from mmseg.models.losses import (
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    LovaszLoss,
)
