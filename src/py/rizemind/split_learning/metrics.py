"""Classification metrics for federated split-learning evaluation.

NumPy-only (no scikit-learn dependency) so it works in the base ``rizemind``
install. All functions are edge-case safe: zero-division yields 0.0, and
metrics that are undefined for a single observed class (ROC-AUC) return
``None`` rather than raising.

The headline entry point is :func:`classification_metrics`, which returns a
plain ``dict`` of JSON-serialisable scalars suitable for writing straight into
a result record.
"""

from __future__ import annotations

import numpy as np

__all__ = ["confusion_counts", "roc_auc_binary", "classification_metrics"]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Return the ``num_classes x num_classes`` confusion matrix (rows=true).

    Args:
        y_true: Integer ground-truth labels.
        y_pred: Integer predicted labels.
        num_classes: Number of classes.

    Returns:
        Integer array ``cm`` with ``cm[t, p]`` = count of true ``t`` predicted
        ``p``.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Binary ROC-AUC via the rank (Mann-Whitney U) statistic.

    Args:
        y_true: Ground-truth labels in {0, 1}.
        y_score: Score / probability for the positive class.

    Returns:
        ROC-AUC in ``[0, 1]``, or ``None`` if only one class is present (AUC
        undefined).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # Average ranks (handles ties correctly).
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    *,
    num_classes: int | None = None,
    positive_class: int = 1,
) -> dict:
    """Compute aggregate classification metrics from labels and predictions.

    Macro-averages precision/recall/F1 across classes; also reports the
    positive-class precision/recall/F1 for binary problems. ROC-AUC is computed
    only when ``y_score`` is given and the problem is binary with both classes
    present.

    Args:
        y_true: Integer ground-truth labels.
        y_pred: Integer predicted labels.
        y_score: Optional positive-class scores/probabilities (binary only),
            for ROC-AUC.
        num_classes: Number of classes; inferred from the data if omitted.
        positive_class: Which class is "positive" for binary P/R/F1.

    Returns:
        A dict of JSON-serialisable scalars: ``accuracy``, ``balanced_accuracy``,
        ``precision_macro``, ``recall_macro``, ``f1_macro``, binary
        ``precision``/``recall``/``f1`` (when 2 classes), ``roc_auc`` (or None),
        ``n_samples``, ``num_classes``, ``confusion`` (flat row-major list), and
        ``class_distribution`` (true-label counts).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = int(y_true.shape[0])
    if num_classes is None:
        max_label = int(max(y_true.max(initial=0), y_pred.max(initial=0)))
        num_classes = max_label + 1
    num_classes = max(num_classes, 2)

    cm = confusion_counts(y_true, y_pred, num_classes)
    correct = int(np.trace(cm))
    accuracy = _safe_div(correct, n)

    precisions, recalls, f1s, recalls_present = [], [], [], []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        if support > 0:
            recalls_present.append(rec)

    balanced_accuracy = float(np.mean(recalls_present)) if recalls_present else 0.0

    out: dict = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "n_samples": n,
        "num_classes": int(num_classes),
        "confusion": cm.flatten().tolist(),
        "class_distribution": [int((y_true == c).sum()) for c in range(num_classes)],
    }
    if num_classes == 2:
        out["precision"] = precisions[positive_class]
        out["recall"] = recalls[positive_class]
        out["f1"] = f1s[positive_class]
        out["roc_auc"] = (
            roc_auc_binary(y_true, y_score) if y_score is not None else None
        )
    return out
