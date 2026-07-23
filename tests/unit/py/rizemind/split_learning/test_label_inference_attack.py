"""Unit tests for the label-inference attack benchmark.

Verify split isolation (shadow/eval are disjoint), that a known label-correlated
gradient signal is recovered, that pure-noise gradients stay near chance,
majority/chance references, and that labels are never echoed as features.
"""

import numpy as np
import pytest
from rizemind.split_learning.label_inference_attack import (
    AttackSplit,
    evaluate_label_inference,
    gradient_row_features,
    split_indices,
)


def test_split_is_disjoint_and_covers_all():
    split = split_indices(100, seed=3, shadow_fraction=0.5)
    assert isinstance(split, AttackSplit)
    s, e = set(split.shadow_idx.tolist()), set(split.eval_idx.tolist())
    assert s.isdisjoint(e)
    assert s | e == set(range(100))
    assert split.seed == 3


def test_split_seed_reproducible():
    a = split_indices(50, seed=7)
    b = split_indices(50, seed=7)
    assert np.array_equal(a.shadow_idx, b.shadow_idx)
    assert np.array_equal(a.eval_idx, b.eval_idx)


def test_split_rejects_tiny_n():
    with pytest.raises(ValueError):
        split_indices(1, seed=0)


def test_features_shape_and_coordinator_visible():
    g = np.random.RandomState(0).randn(10, 6)
    feats = gradient_row_features(g)
    assert feats.shape == (10, 12)
    assert np.all(np.isfinite(feats))


def test_attack_recovers_label_correlated_signal():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    g = rng.normal(scale=0.1, size=(n, 8))
    # Encode the label in the sign of coordinate 0 (mimics CE cut-gradient sign).
    g[:, 0] += np.where(y == 1, 1.0, -1.0)
    res = evaluate_label_inference(g, y, seed=0)
    assert res["attack_a"]["roc_auc"] > 0.95
    assert res["attack_b"]["roc_auc"] > 0.95
    # shadow/eval disjoint sizes recorded
    assert res["n_shadow"] + res["n_eval"] == n


def test_pure_noise_stays_near_chance():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, n)
    g = rng.normal(size=(n, 8))  # independent of y
    res = evaluate_label_inference(g, y, seed=0)
    # Learned attack AUC should be close to 0.5 (no signal). Allow slack.
    assert abs(res["attack_b"]["roc_auc"] - 0.5) < 0.12


def test_references_present():
    rng = np.random.default_rng(2)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)  # 30% positive
    g = rng.normal(size=(n, 4))
    res = evaluate_label_inference(g, y, seed=0)
    refs = res["references"]
    assert refs["chance_accuracy"] == 0.5
    assert refs["majority_class"] == 0
    assert refs["majority_accuracy"] == pytest.approx(
        max(refs["eval_positive_rate"], 1 - refs["eval_positive_rate"])
    )


def test_labels_must_be_binary():
    g = np.random.RandomState(0).randn(10, 4)
    y = np.array([0, 1, 2] * 3 + [0])
    with pytest.raises(ValueError):
        evaluate_label_inference(g, y, seed=0)


def test_metrics_are_scalar_only():
    rng = np.random.default_rng(3)
    n = 120
    y = rng.integers(0, 2, n)
    g = rng.normal(size=(n, 5))
    res = evaluate_label_inference(g, y, seed=0)
    for attack in ("attack_a", "attack_b"):
        for k, v in res[attack].items():
            assert v is None or np.isscalar(v) or isinstance(v, str), (attack, k)
