from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

# labels = torch.cat(self.epoch_labels_preds["labels"])
# preds = torch.cat(self.epoch_labels_preds["preds"])
# for metric_name, metric_fn in self._metrics.items():
#     for c_idx, class_name in enumerate(self.classes):
#         if metric_name == "bs":
#             epoch_metrics[f"{class_name}-{metric_name}"] = metric_fn(
#                 y_true=labels[:, c_idx], y_prob=preds[:, c_idx]
#             )
#         else:
#             epoch_metrics[f"{class_name}-{metric_name}"] = metric_fn(
#                 y_true=labels[:, c_idx], y_score=preds[:, c_idx]
#             )
#     epoch_metrics[f"{metric_name}-macro"] = np.mean(
#         [
#             epoch_metrics[f"{class_name}-{metric_name}"]
#             for class_name in self.classes
#         ]
#     )
