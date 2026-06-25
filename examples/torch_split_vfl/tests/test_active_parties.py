"""Tests for genuine federated feature ablations + seed + telemetry wiring."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from rizemind.split_learning.serialization import tensor_to_parameters
from rizemind.split_learning.telemetry import RunTelemetry
from torch_split_vfl.server import make_on_train_step
from torch_split_vfl.task import (
    build_bottom_model,
    build_server_top,
    get_dataset_spec,
    load_tabular,
    make_client_train_partition,
    make_server_train_labels,
    parse_active_parties,
    party_feature_dim,
    resolve_active_groups,
)


# --------------------------------------------------------------------------
# parse_active_parties
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("", None),
        ("all", None),
        (None, None),
        ("0", (0,)),
        ("1", (1,)),
        ("0,1", (0, 1)),
        (" 0 , 1 ", (0, 1)),
    ],
)
def test_parse_active_parties(value, expected):
    assert parse_active_parties(value) == expected


def test_parse_active_parties_invalid():
    with pytest.raises(ValueError):
        parse_active_parties("a,b")


# --------------------------------------------------------------------------
# resolve_active_groups — Adult (tabular, 2 groups)
# --------------------------------------------------------------------------
def _adult():
    return get_dataset_spec("adult")


def test_both_parties_default():
    assert resolve_active_groups(_adult(), 2, None) == [0, 1]


def test_party0_only():
    assert resolve_active_groups(_adult(), 1, (0,)) == [0]


def test_party1_only():
    assert resolve_active_groups(_adult(), 1, (1,)) == [1]


def test_invalid_out_of_range():
    with pytest.raises(ValueError):
        resolve_active_groups(_adult(), 1, (5,))


def test_invalid_count_mismatch():
    with pytest.raises(ValueError):
        resolve_active_groups(_adult(), 2, (0,))  # 1 group but K=2


def test_invalid_duplicate():
    with pytest.raises(ValueError):
        resolve_active_groups(_adult(), 2, (0, 0))


def test_invalid_empty():
    with pytest.raises(ValueError):
        resolve_active_groups(_adult(), 1, ())


def test_image_subset_rejected():
    mnist = get_dataset_spec("mnist")
    with pytest.raises(ValueError):
        resolve_active_groups(mnist, 4, (0, 1))  # strip subset not allowed


# --------------------------------------------------------------------------
# Feature dims + server input width per ablation
# --------------------------------------------------------------------------
def test_party_feature_dims_match_groups():
    spec = _adult()
    pt = load_tabular(spec)
    d0, d1 = pt.party_dims
    # party0-only: client 0 owns group 0 -> dim d0
    assert party_feature_dim(spec, 0, 1, (0,)) == d0
    # party1-only: client 0 owns group 1 -> dim d1
    assert party_feature_dim(spec, 0, 1, (1,)) == d1
    # both: client 0 -> d0, client 1 -> d1
    assert party_feature_dim(spec, 0, 2, None) == d0
    assert party_feature_dim(spec, 1, 2, None) == d1


def test_server_input_width_scales_with_active_count():
    spec = _adult()
    assert build_server_top(spec, num_clients=1, hidden_dim=64).fc1.in_features == 64
    assert build_server_top(spec, num_clients=2, hidden_dim=64).fc1.in_features == 128


def test_party1_only_bottom_input():
    spec = _adult()
    d1 = load_tabular(spec).party_dims[1]
    bottom = build_bottom_model(spec, 0, 1, hidden_dim=32, active_parties=(1,))
    assert bottom.fc.in_features == d1


# --------------------------------------------------------------------------
# Sample alignment: same shuffle seed => identical batch indices
# --------------------------------------------------------------------------
def test_sample_alignment_same_seed():
    spec = _adult()
    p_client = make_client_train_partition(
        spec, 0, 2, batch_size=16, max_train_samples=500, shuffle_seed=7
    )
    p_labels = make_server_train_labels(
        spec, batch_size=16, max_train_samples=500, shuffle_seed=7
    )
    # Same permutation source -> same row order for a given step.
    perm_c = p_client._permutation_for_epoch(0)
    perm_l = p_labels._permutation_for_epoch(0)
    assert torch.equal(perm_c, perm_l)


def test_different_seed_changes_order():
    spec = _adult()
    a = make_server_train_labels(spec, 16, 500, shuffle_seed=1)._permutation_for_epoch(0)
    b = make_server_train_labels(spec, 16, 500, shuffle_seed=2)._permutation_for_epoch(0)
    assert not torch.equal(a, b)


# --------------------------------------------------------------------------
# Label isolation: client partitions never expose the label tensor
# --------------------------------------------------------------------------
def test_label_isolation():
    spec = _adult()
    pt = load_tabular(spec)
    # A client partition's feature width equals an encoded feature group, never
    # 1 (the label column is held only by the server labels partition).
    client = make_client_train_partition(spec, 0, 1, 16, 200, active_parties=(0,))
    assert client.features.shape[1] == pt.party_dims[0]
    labels = make_server_train_labels(spec, 16, 200)
    assert labels.features.dim() == 1  # label vector lives only at the server


# --------------------------------------------------------------------------
# Telemetry: on_train_step records activation + gradient bytes
# --------------------------------------------------------------------------
def test_on_train_step_records_telemetry():
    spec = _adult()
    tail = build_server_top(spec, num_clients=2, hidden_dim=8)
    opt = torch.optim.SGD(tail.parameters(), lr=0.1)
    labels = make_server_train_labels(spec, batch_size=4, max_train_samples=64)
    telem = RunTelemetry()
    on_step = make_on_train_step(
        tail, opt, nn.CrossEntropyLoss(), labels, [], demo=False, telemetry=telem
    )
    # Two fake activations (B=4, H=8), tagged pid 0 and 1.
    acts = [torch.randn(4, 8, requires_grad=True) for _ in range(2)]
    ordered = [(i, f"cid{i}", tensor_to_parameters(a.detach())) for i, a in enumerate(acts)]
    grad_store, loss = on_step(0, ordered)
    assert set(grad_store) == {"cid0", "cid1"}
    d = telem.as_dict()
    # 4*8*4 bytes float32 per activation, x2 parties
    assert d["cumulative_activation_bytes"] == 2 * (4 * 8 * 4)
    assert d["cumulative_gradient_bytes"] == 2 * (4 * 8 * 4)
    assert d["n_steps"] == 1
    assert "server_forward" in d["timings_s"]


def test_metrics_serializable_json():
    import json

    from rizemind.split_learning.metrics import classification_metrics

    m = classification_metrics(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]),
        y_score=np.array([0.2, 0.9, 0.4, 0.1]),
    )
    json.dumps(m)  # must not raise
    assert "f1" in m and "roc_auc" in m and "confusion" in m
