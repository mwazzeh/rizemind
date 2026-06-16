"""Tests for rizemind.split_learning.metrics."""

import numpy as np

from rizemind.split_learning.metrics import (
    classification_metrics,
    confusion_counts,
    roc_auc_binary,
)


def test_perfect_binary():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, y, y_score=np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion"] == [2, 0, 0, 2]
    assert m["class_distribution"] == [2, 2]
    assert m["n_samples"] == 4


def test_known_confusion_and_prf():
    # 2 classes; construct a known confusion matrix.
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])  # tp=2, fn=1, tn=2, fp=1
    m = classification_metrics(y_true, y_pred)
    assert m["confusion"] == [2, 1, 1, 2]  # rows=true [tn,fp ; fn,tp]
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert abs(m["f1"] - 2 / 3) < 1e-9
    assert m["accuracy"] == 4 / 6


def test_zero_division_safe():
    # All predicted negative -> positive precision/recall divide by zero.
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_single_class_roc_auc_none():
    y_true = np.array([1, 1, 1])
    assert roc_auc_binary(y_true, np.array([0.2, 0.5, 0.9])) is None
    m = classification_metrics(y_true, y_true, y_score=np.array([0.2, 0.5, 0.9]))
    assert m["roc_auc"] is None


def test_roc_auc_with_ties():
    y_true = np.array([0, 0, 1, 1])
    # tied scores between a neg and a pos -> AUC 0.5 region handling
    auc = roc_auc_binary(y_true, np.array([0.5, 0.5, 0.5, 0.9]))
    assert 0.0 <= auc <= 1.0


def test_confusion_counts_shape():
    cm = confusion_counts(np.array([0, 1, 2]), np.array([0, 1, 1]), num_classes=3)
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == 3


def test_multiclass_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 0])
    m = classification_metrics(y_true, y_pred, num_classes=3)
    assert m["num_classes"] == 3
    assert "precision_macro" in m and "f1_macro" in m
    assert "roc_auc" not in m  # only for binary
