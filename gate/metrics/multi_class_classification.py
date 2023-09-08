import sklearn.metrics as sk_metrics


def average_precision_score(y_true, y_pred, **kwargs):
    return sk_metrics.average_precision_score(y_true, y_pred, **kwargs)


def brier_score_loss(y_true, y_pred, **kwargs):
    return sk_metrics.brier_score_loss(y_true, y_pred, **kwargs)


def roc_auc_score(y_true, y_pred, **kwargs):
    return sk_metrics.roc_auc_score(y_true, y_pred, **kwargs)
