"""Tests for the UCI Adult tabular VFL task.

These run fully offline on a tiny synthetic table (no dataset download): they
exercise the deterministic preprocessing, the vertical column split, sample
alignment, label safety, and a short end-to-end training step.
"""

import pytest
import torch
import torch.nn as nn
from torch_split_vfl.task import (
    BottomMLP,
    ServerTopMLP,
    TabularSpec,
    deterministic_split,
    get_dataset_spec,
    party_feature_dim,
    preprocess_tabular,
    validate_tabular_spec,
)

# A small Adult-shaped spec: party 0 = person columns, party 1 = work columns.
TSPEC = TabularSpec(
    hf_path="unused-in-tests",
    label_column="y",
    positive_label="yes",
    numerical=("age", "hours"),
    categorical=("sex", "country"),
    party_columns=(("age", "sex"), ("hours", "country")),
)

# sex vocab = {F, M} (size 2); country vocab = {?, CA, US} (size 3, '?' is a
# normal category). So party 0 dim = 1 (age) + 2 (sex) = 3; party 1 dim =
# 1 (hours) + 3 (country) = 4.
TRAIN = {
    "age": [20, 30, 40, 50],
    "hours": [40, 35, 50, 45],
    "sex": ["M", "F", "M", "F"],
    "country": ["US", "CA", "US", "?"],  # '?' = missing, kept as a category
    "y": ["yes", "no", "yes", "no"],
}
# Test set includes an unseen category ("BR") to check it maps to all-zeros.
TEST = {
    "age": [25, 60],
    "hours": [38, 42],
    "sex": ["F", "M"],
    "country": ["US", "BR"],
    "y": ["no", "yes"],
}


def _pre():
    return preprocess_tabular(TRAIN, TEST, TSPEC)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_adult_registered_as_tabular():
    spec = get_dataset_spec("adult")
    assert spec.kind == "tabular"
    assert spec.num_classes == 2
    assert spec.tabular is not None
    # Realistic 2-party split, label held out.
    assert len(spec.tabular.party_columns) == 2
    for cols in spec.tabular.party_columns:
        assert spec.tabular.label_column not in cols


# ---------------------------------------------------------------------------
# Feature partition shapes
# ---------------------------------------------------------------------------


def test_party_dims_and_shapes():
    pre = _pre()
    assert pre.party_dims == [3, 4]
    assert pre.party_train[0].shape == (4, 3)
    assert pre.party_train[1].shape == (4, 4)
    assert pre.party_test[0].shape == (2, 3)
    assert pre.party_test[1].shape == (2, 4)


def test_party_feature_dim_rejects_wrong_client_count():
    spec = get_dataset_spec("adult")
    # Adult defines 2 parties; asking for 3 must raise before any data load.
    with pytest.raises(ValueError):
        party_feature_dim(spec, 0, 3)


def test_one_hot_and_standardization():
    pre = _pre()
    # Numerical age column (col 0 of party 0) is standardized: train mean ~0.
    age_col = pre.party_train[0][:, 0]
    assert abs(age_col.mean().item()) < 1e-5
    assert abs(age_col.std(unbiased=False).item() - 1.0) < 1e-5
    # sex vocab is sorted [F, M]; row 0 is "M" -> one-hot [0, 1].
    assert pre.party_train[0][0, 1:].tolist() == [0.0, 1.0]
    # Unseen test category "BR" (row 1, party 1 country block) -> all zeros.
    assert pre.party_test[1][1, 1:].tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Sample alignment
# ---------------------------------------------------------------------------


def test_all_parties_have_same_sample_count():
    pre = _pre()
    n_train = pre.train_labels.shape[0]
    n_test = pre.test_labels.shape[0]
    assert n_train == 4 and n_test == 2
    for t in pre.party_train:
        assert t.shape[0] == n_train
    for t in pre.party_test:
        assert t.shape[0] == n_test


def test_labels_align_with_rows():
    pre = _pre()
    # "yes" -> 1, anything else -> 0, in original row order.
    assert pre.train_labels.tolist() == [1, 0, 1, 0]
    assert pre.test_labels.tolist() == [0, 1]


def test_deterministic_split_is_reproducible_and_disjoint():
    table = {"a": list(range(20)), "y": ["yes"] * 20}
    tr1, te1 = deterministic_split(table, test_fraction=0.25, seed=7)
    tr2, te2 = deterministic_split(table, test_fraction=0.25, seed=7)
    assert tr1["a"] == tr2["a"] and te1["a"] == te2["a"]  # reproducible
    assert len(te1["a"]) == 5 and len(tr1["a"]) == 15
    assert set(tr1["a"]).isdisjoint(te1["a"])  # no row in both
    assert set(tr1["a"]) | set(te1["a"]) == set(range(20))  # full coverage
    # A different seed gives a different partition.
    _, te3 = deterministic_split(table, test_fraction=0.25, seed=8)
    assert te3["a"] != te1["a"]


# ---------------------------------------------------------------------------
# Label safety (no leakage)
# ---------------------------------------------------------------------------


def test_no_label_leakage_into_party_features():
    pre = _pre()
    # Flip every label; party features must be byte-for-byte identical, proving
    # they are computed without ever reading the label column.
    flipped = {**TRAIN, "y": ["no", "yes", "no", "yes"]}
    pre2 = preprocess_tabular(flipped, TEST, TSPEC)
    for a, b in zip(pre.party_train, pre2.party_train):
        assert torch.equal(a, b)
    for a, b in zip(pre.party_test, pre2.party_test):
        assert torch.equal(a, b)


def test_validate_rejects_label_in_party():
    leaky = TabularSpec(
        hf_path="x",
        label_column="y",
        positive_label="yes",
        numerical=("age",),
        categorical=("sex",),
        party_columns=(("age", "y"), ("sex",)),  # 'y' leaked into party 0
    )
    with pytest.raises(ValueError):
        validate_tabular_spec(leaky)


def test_validate_rejects_duplicate_and_unknown_columns():
    dup = TabularSpec(
        hf_path="x", label_column="y", positive_label="yes",
        numerical=("age",), categorical=("sex",),
        party_columns=(("age",), ("age",)),  # age assigned twice
    )
    with pytest.raises(ValueError):
        validate_tabular_spec(dup)
    unknown = TabularSpec(
        hf_path="x", label_column="y", positive_label="yes",
        numerical=("age",), categorical=("sex",),
        party_columns=(("age", "ghost"), ("sex",)),  # 'ghost' not declared
    )
    with pytest.raises(ValueError):
        validate_tabular_spec(unknown)


# ---------------------------------------------------------------------------
# End-to-end smoke training step
# ---------------------------------------------------------------------------


def test_short_training_run_reduces_loss():
    """A few VFL steps over the synthetic table run and reduce the loss."""
    torch.manual_seed(0)  # keep ReLU units live + result deterministic
    pre = _pre()
    bottoms = [BottomMLP(d, hidden_dim=8) for d in pre.party_dims]
    tail = ServerTopMLP(num_clients=2, hidden_dim=8, num_classes=2)
    params = [p for b in bottoms for p in b.parameters()] + list(tail.parameters())
    opt = torch.optim.SGD(params, lr=0.1)
    criterion = nn.CrossEntropyLoss()
    y = pre.train_labels

    first_loss = last_loss = None
    weights_before = bottoms[0].fc.weight.detach().clone()
    for _ in range(25):
        opt.zero_grad()
        joint = torch.cat([b(pre.party_train[i]) for i, b in enumerate(bottoms)], dim=1)
        loss = criterion(tail(joint), y)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss

    assert torch.isfinite(torch.tensor(last_loss))
    assert last_loss < first_loss  # the pipeline actually learns
    # Bottom-model weights moved (gradients flowed back through the cut).
    assert not torch.equal(weights_before, bottoms[0].fc.weight)
