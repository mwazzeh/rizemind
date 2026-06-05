import pytest
import torch
from torch_split_vfl.task import (
    BottomMLP,
    ServerTopMLP,
    VerticalPartition,
    build_bottom_model,
    build_server_top,
    get_dataset_spec,
    strip_bounds,
    strip_feature_dim,
)

# ---------------------------------------------------------------------------
# strip layout
# ---------------------------------------------------------------------------


def test_strip_bounds_divides_evenly():
    # 28 cols / 2 clients
    assert strip_bounds(0, 2, 28) == (0, 14)
    assert strip_bounds(1, 2, 28) == (14, 28)
    # 28 / 4
    assert strip_bounds(0, 4, 28) == (0, 7)
    assert strip_bounds(3, 4, 28) == (21, 28)


def test_strip_bounds_last_client_absorbs_remainder():
    # 28 / 3 = base 9; last gets 28 - 18 = 10
    assert strip_bounds(0, 3, 28) == (0, 9)
    assert strip_bounds(1, 3, 28) == (9, 18)
    assert strip_bounds(2, 3, 28) == (18, 28)


def test_strip_bounds_validates_inputs():
    with pytest.raises(ValueError):
        strip_bounds(-1, 2, 28)
    with pytest.raises(ValueError):
        strip_bounds(2, 2, 28)
    with pytest.raises(ValueError):
        strip_bounds(0, 0, 28)
    with pytest.raises(ValueError):
        strip_bounds(0, 30, 28)


def test_strip_feature_dim_matches_bounds():
    spec = get_dataset_spec("mnist")
    # 1 channel * 28 rows * 14 cols
    assert strip_feature_dim(spec, 0, 2) == 1 * 28 * 14


# ---------------------------------------------------------------------------
# VerticalPartition step alignment
# ---------------------------------------------------------------------------


def test_step_to_epoch_batch_wraps_after_epoch():
    p = VerticalPartition(features=torch.zeros(10, 1), batch_size=3)
    # num_batches = ceil(10/3) = 4
    assert p.num_batches == 4
    assert p.step_to_epoch_batch(0) == (0, 0)
    assert p.step_to_epoch_batch(3) == (0, 3)
    assert p.step_to_epoch_batch(4) == (1, 0)


def test_get_batch_for_step_is_deterministic_across_partitions():
    # Two partitions with identical sample order should hit the same indices.
    features_a = torch.arange(20, dtype=torch.float32).view(20, 1)
    features_b = torch.arange(20, dtype=torch.float32).view(20, 1) * 100
    pa = VerticalPartition(features=features_a, batch_size=5)
    pb = VerticalPartition(features=features_b, batch_size=5)

    # Same shuffle seed means identical permutation; b is a*100 → b/100 == a.
    for step in range(8):
        ba = pa.get_batch_for_step(step)
        bb = pb.get_batch_for_step(step)
        assert torch.allclose(ba * 100, bb)


def test_get_batch_for_step_last_batch_may_be_short():
    features = torch.arange(11, dtype=torch.float32).view(11, 1)
    p = VerticalPartition(features=features, batch_size=4)
    # batches: 4, 4, 3
    sizes = [p.get_batch_for_step(s).shape[0] for s in range(p.num_batches)]
    assert sizes == [4, 4, 3]


def test_get_batch_for_step_rejects_negative():
    p = VerticalPartition(features=torch.zeros(4, 1), batch_size=2)
    with pytest.raises(ValueError):
        p.get_batch_for_step(-1)


def test_get_batch_for_step_is_aligned_across_four_parties():
    """Four parties + a server — all derive the same (epoch, batch) and same ids
    from a given sl_step, because the shuffle seed is shared.
    """
    n = 16
    # Encode the sample id directly in the feature so we can compare.
    ids = torch.arange(n, dtype=torch.float32).view(n, 1)
    parties = [
        VerticalPartition(features=ids * scale, batch_size=4)
        for scale in (1.0, 7.0, -3.0, 11.0)
    ]
    server_labels = VerticalPartition(features=ids.clone().view(-1), batch_size=4)

    for step in range(parties[0].num_batches * 2):  # two full epochs
        per_party_ids = [
            p.get_batch_for_step(step) / scale
            for p, scale in zip(parties, [1.0, 7.0, -3.0, 11.0])
        ]
        server_batch = server_labels.get_batch_for_step(step)
        # All parties report the same sample ids (modulo encoding scale).
        for a, b in zip(per_party_ids, per_party_ids[1:]):
            torch.testing.assert_close(a, b)
        # Server labels (held separately) hit the same ids.
        torch.testing.assert_close(per_party_ids[0].view(-1), server_batch)


def test_step_to_epoch_batch_advances_through_full_epoch():
    p = VerticalPartition(features=torch.zeros(20, 1), batch_size=4)
    # 5 batches per epoch.
    pairs = [p.step_to_epoch_batch(s) for s in range(11)]
    assert pairs == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
    ]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_bottom_mlp_output_shape():
    m = BottomMLP(in_features=10, hidden_dim=4)
    x = torch.zeros(3, 10)
    out = m(x)
    assert out.shape == (3, 4)


def test_server_top_input_size_matches_num_clients_times_hidden():
    spec = get_dataset_spec("mnist")
    tail = build_server_top(spec, num_clients=3, hidden_dim=5)
    assert isinstance(tail, ServerTopMLP)
    # joint = (B, 3*5) → fc1 in_features must equal 15.
    assert tail.fc1.in_features == 15
    assert tail.fc2.out_features == spec.num_classes


def test_build_bottom_model_uses_correct_strip_dim():
    spec = get_dataset_spec("mnist")
    m = build_bottom_model(spec, partition_id=0, num_clients=2, hidden_dim=8)
    # strip 0 of 2: 1 * 28 * 14 = 392
    assert m.fc.in_features == 392
    assert m.fc.out_features == 8
