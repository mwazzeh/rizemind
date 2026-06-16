"""Label-inference attack benchmark on coordinator-visible cut gradients.

This implements the *attacker* side of the Phase-4 study: given per-sample cut
gradient rows (exactly what an honest-but-curious coordinator sees) and the true
labels (used **only** to score the attack, never as an attack feature), measure
how recoverable the labels are. Two attackers are provided and reported
separately (see ``THREAT_MODEL.md``):

* **Attack A — no-training heuristic.** Defensible per-row summary statistics
  (L2 norm, signed mean, sign proportions, extrema, low-order moments). The most
  label-correlated single statistic and its threshold/orientation are chosen on
  the *attack-training* split only, then applied to held-out data.
* **Attack B — learned shadow attack.** A logistic-regression classifier (plain
  NumPy, no scikit-learn dependency) fit on a disjoint *shadow* split of
  (gradient-row, label) pairs, evaluated on held-out data.

Split discipline (enforced): the attack is fit on a shadow split and evaluated on
a disjoint held-out split; labels are never used as input features; the same
preprocessing is applied to baseline and protected gradients. Raw labels and raw
gradients are never returned — only aggregate attack metrics.

This module is NumPy-only so it imports in the base ``rizemind`` install.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rizemind.split_learning.metrics import roc_auc_binary

__all__ = [
    "AttackSplit",
    "gradient_row_features",
    "split_indices",
    "evaluate_label_inference",
]


@dataclass(frozen=True)
class AttackSplit:
    """Disjoint shadow-train / attack-eval index split (recorded for provenance).

    Attributes:
        shadow_idx: Indices used to fit/calibrate the attack.
        eval_idx: Held-out indices used to score the attack.
        seed: Seed of the permutation that produced the split.
        shadow_fraction: Fraction of samples assigned to the shadow split.
    """

    shadow_idx: np.ndarray
    eval_idx: np.ndarray
    seed: int
    shadow_fraction: float


def split_indices(n: int, *, seed: int, shadow_fraction: float = 0.5) -> AttackSplit:
    """Produce a disjoint shadow/eval split of ``range(n)`` from a seeded perm.

    Args:
        n: Number of samples.
        seed: Permutation seed (recorded).
        shadow_fraction: Fraction assigned to the shadow (attack-training) split.

    Returns:
        An :class:`AttackSplit`.

    Raises:
        ValueError: If ``n < 2`` or ``shadow_fraction`` is not in ``(0, 1)``.
    """
    if n < 2:
        raise ValueError(f"need at least 2 samples to split, got {n}")
    if not (0.0 < shadow_fraction < 1.0):
        raise ValueError(f"shadow_fraction must be in (0,1), got {shadow_fraction}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_shadow = max(1, min(n - 1, int(round(n * shadow_fraction))))
    return AttackSplit(
        shadow_idx=perm[:n_shadow],
        eval_idx=perm[n_shadow:],
        seed=int(seed),
        shadow_fraction=float(shadow_fraction),
    )


#: Names of the engineered per-row statistics used by Attack A.
_STAT_NAMES = (
    "l2_norm",
    "l1_norm",
    "signed_mean",
    "std",
    "frac_pos",
    "frac_neg",
    "max",
    "min",
    "mean_pos",
    "mean_neg",
    "third_moment",
    "fourth_moment",
)


def gradient_row_features(grad: np.ndarray) -> np.ndarray:
    """Map each ``(D,)`` gradient row to a fixed vector of summary statistics.

    All quantities are computable by the coordinator from the released gradient
    rows alone (no labels). Returns a ``(B, len(_STAT_NAMES))`` float matrix.
    """
    g = np.asarray(grad, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"expected a 2-D (B, D) gradient, got shape {g.shape}")
    d = g.shape[1]
    mean = g.mean(axis=1)
    std = g.std(axis=1)
    centered = g - mean[:, None]
    safe_std = np.where(std > 0, std, 1.0)
    third = (centered**3).mean(axis=1) / (safe_std**3)
    fourth = (centered**4).mean(axis=1) / (safe_std**4)
    pos = g > 0
    neg = g < 0
    n_pos = pos.sum(axis=1)
    n_neg = neg.sum(axis=1)
    mean_pos = np.where(n_pos > 0, (g * pos).sum(axis=1) / np.maximum(n_pos, 1), 0.0)
    mean_neg = np.where(n_neg > 0, (g * neg).sum(axis=1) / np.maximum(n_neg, 1), 0.0)
    feats = np.stack(
        [
            np.sqrt((g * g).sum(axis=1)),  # l2
            np.abs(g).sum(axis=1),  # l1
            mean,  # signed mean
            std,
            n_pos / d,
            n_neg / d,
            g.max(axis=1),
            g.min(axis=1),
            mean_pos,
            mean_neg,
            third,
            fourth,
        ],
        axis=1,
    )
    return feats


def _binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    auc = roc_auc_binary(y_true, y_score) if y_score is not None else None
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": (float(auc) if auc is not None else None),
    }


def _standardize(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return tuple((a - mean) / std for a in (train, *arrays))


def _logreg_fit(
    x: np.ndarray, y: np.ndarray, *, epochs: int = 300, lr: float = 0.5, l2: float = 1e-3
) -> np.ndarray:
    """Plain full-batch logistic-regression fit (NumPy). Returns weights w (D+1,)."""
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    w = np.zeros(d + 1)
    yf = y.astype(np.float64)
    for _ in range(epochs):
        z = xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = xb.T @ (p - yf) / n + l2 * np.concatenate([w[:-1], [0.0]])
        w -= lr * grad
    return w


def _logreg_score(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    z = xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _attack_a(
    stats: np.ndarray, y: np.ndarray, split: AttackSplit
) -> dict:
    """No-training heuristic: pick the best single statistic + threshold on the
    shadow split, evaluate on the eval split. Threshold orientation is chosen
    from shadow data only."""
    s_tr, y_tr = stats[split.shadow_idx], y[split.shadow_idx]
    s_te, y_te = stats[split.eval_idx], y[split.eval_idx]
    best = None
    for j in range(stats.shape[1]):
        col_tr = s_tr[:, j]
        # Candidate thresholds = midpoints between sorted unique train values.
        uniq = np.unique(col_tr)
        if uniq.size < 2:
            continue
        cands = (uniq[:-1] + uniq[1:]) / 2.0
        # Cap candidate count for speed.
        if cands.size > 64:
            cands = np.quantile(col_tr, np.linspace(0.02, 0.98, 64))
        for thr in cands:
            for orient in (1, -1):
                pred_tr = ((col_tr > thr).astype(int) if orient == 1
                           else (col_tr <= thr).astype(int))
                # balanced accuracy on the shadow split as the selection score
                m = _binary_metrics(y_tr, pred_tr, None)["balanced_accuracy"]
                if best is None or m > best[0]:
                    best = (m, j, float(thr), orient)
    if best is None:
        # Degenerate (no usable statistic): predict majority.
        maj = int(round(y_tr.mean()))
        pred = np.full(len(y_te), maj)
        out = _binary_metrics(y_te, pred, None)
        out.update(statistic="<none>", orientation=0)
        return out
    _score, j, thr, orient = best
    col_te = s_te[:, j]
    pred_te = ((col_te > thr).astype(int) if orient == 1
               else (col_te <= thr).astype(int))
    # Continuous score (oriented) for ROC-AUC.
    score_te = col_te if orient == 1 else -col_te
    out = _binary_metrics(y_te, pred_te, score_te)
    out.update(statistic=_STAT_NAMES[j], orientation=int(orient))
    return out


def _attack_b(grad: np.ndarray, y: np.ndarray, split: AttackSplit) -> dict:
    """Learned shadow attack: logistic regression on standardized raw gradient
    rows, fit on the shadow split, evaluated on the eval split."""
    x_tr_raw, y_tr = grad[split.shadow_idx], y[split.shadow_idx]
    x_te_raw, y_te = grad[split.eval_idx], y[split.eval_idx]
    x_tr, x_te = _standardize(x_tr_raw.astype(np.float64), x_te_raw.astype(np.float64))
    if len(np.unique(y_tr)) < 2:
        maj = int(round(y_tr.mean()))
        pred = np.full(len(y_te), maj)
        return _binary_metrics(y_te, pred, None)
    w = _logreg_fit(x_tr, y_tr)
    score_te = _logreg_score(w, x_te)
    pred_te = (score_te >= 0.5).astype(int)
    return _binary_metrics(y_te, pred_te, score_te)


def _references(y_eval: np.ndarray) -> dict:
    """Majority-class and random-chance references on the eval labels."""
    n = len(y_eval)
    p1 = float(y_eval.mean()) if n else 0.0
    maj_class = 1 if p1 >= 0.5 else 0
    maj_acc = max(p1, 1.0 - p1)
    return {
        "majority_class": maj_class,
        "majority_accuracy": float(maj_acc),
        "chance_accuracy": 0.5,
        "chance_balanced_accuracy": 0.5,
        "chance_roc_auc": 0.5,
        "eval_positive_rate": p1,
    }


def evaluate_label_inference(
    grad: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
    shadow_fraction: float = 0.5,
    include_attack_a: bool = True,
    include_attack_b: bool = True,
) -> dict:
    """Run the label-inference benchmark on a gradient matrix.

    Args:
        grad: ``(B, D)`` per-sample cut-gradient matrix (coordinator-visible).
        labels: ``(B,)`` binary ground-truth labels — used only to score the
            attack, never as an input feature.
        seed: Seed for the shadow/eval split (recorded).
        shadow_fraction: Fraction of samples used to fit/calibrate the attack.
        include_attack_a: Run the no-training heuristic attack.
        include_attack_b: Run the learned shadow attack.

    Returns:
        A dict with ``split`` provenance, per-attack metrics under ``attack_a`` /
        ``attack_b``, and ``references`` (majority/chance). Contains only
        aggregate scalars — no labels, no gradient vectors.

    Raises:
        ValueError: If shapes are inconsistent or labels are not binary.
    """
    g = np.asarray(grad, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    if g.ndim != 2:
        raise ValueError(f"grad must be 2-D (B, D), got {g.shape}")
    if y.ndim != 1 or y.shape[0] != g.shape[0]:
        raise ValueError(
            f"labels must be 1-D of length B={g.shape[0]}, got {y.shape}"
        )
    classes = np.unique(y)
    if not np.all(np.isin(classes, (0, 1))):
        raise ValueError(f"labels must be binary {{0,1}}, got classes {classes}")

    split = split_indices(g.shape[0], seed=seed, shadow_fraction=shadow_fraction)
    y_eval = y[split.eval_idx]
    out: dict = {
        "n_total": int(g.shape[0]),
        "n_shadow": int(split.shadow_idx.size),
        "n_eval": int(split.eval_idx.size),
        "split_seed": int(seed),
        "shadow_fraction": float(shadow_fraction),
        "references": _references(y_eval),
    }
    if include_attack_a:
        out["attack_a"] = _attack_a(gradient_row_features(g), y, split)
    if include_attack_b:
        out["attack_b"] = _attack_b(g, y, split)
    return out
