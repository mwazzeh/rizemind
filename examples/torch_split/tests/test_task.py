import torch
from torch_split.task import (
    CachedTrainPartition,
    PartitionConfig,
    deserialize_ndarrays_from_bytes,
    num_server_rounds_for_target_epochs,
    partition_config_from_run_config,
    serialize_ndarrays_to_bytes,
)


def _expected_order(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(size, generator=generator)


def test_cached_train_partition_batches_cover_full_epoch():
    images = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    labels = torch.arange(5, dtype=torch.long)
    partition = CachedTrainPartition(images=images, labels=labels, batch_size=2, shuffle_seed=11)

    batches = [partition.get_batch(batch_idx=i, epoch=0) for i in range(partition.num_batches)]
    epoch_labels = torch.cat([batch_labels for _, batch_labels in batches])

    assert partition.num_batches == 3
    assert batches[-1][0].shape[0] == 1
    assert torch.equal(epoch_labels, labels[_expected_order(size=5, seed=11)])


def test_cached_train_partition_reshuffles_between_epochs():
    images = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    labels = torch.arange(6, dtype=torch.long)
    partition = CachedTrainPartition(images=images, labels=labels, batch_size=2, shuffle_seed=23)

    epoch0_labels = torch.cat(
        [partition.get_batch(batch_idx=i, epoch=0)[1] for i in range(partition.num_batches)]
    )
    epoch1_labels = torch.cat(
        [partition.get_batch(batch_idx=i, epoch=1)[1] for i in range(partition.num_batches)]
    )

    assert torch.equal(epoch0_labels, labels[_expected_order(size=6, seed=23)])
    assert torch.equal(epoch1_labels, labels[_expected_order(size=6, seed=24)])
    assert not torch.equal(epoch0_labels, epoch1_labels)


def test_partition_config_from_run_config_supports_dirichlet():
    config = partition_config_from_run_config({
        "partitioner": "dirichlet",
        "partition-seed": 7,
        "dirichlet-alpha": 0.3,
        "dirichlet-min-partition-size": 12,
        "dirichlet-self-balancing": True,
    })

    assert config == PartitionConfig(
        kind="dirichlet",
        seed=7,
        dirichlet_alpha=0.3,
        dirichlet_min_partition_size=12,
        dirichlet_self_balancing=True,
    )


def test_num_server_rounds_for_target_epochs_uses_max_partition_size():
    rounds = num_server_rounds_for_target_epochs(
        target_epochs=1.5,
        train_batch_counts=(8, 10, 3),
        num_rounds_per_step=2,
    )

    assert rounds == 30


def test_ndarray_bytes_round_trip_preserves_order():
    payload = serialize_ndarrays_to_bytes([
        torch.arange(6, dtype=torch.float32).reshape(2, 3).numpy(),
        torch.tensor([5, 4, 3], dtype=torch.int64).numpy(),
    ])

    restored = deserialize_ndarrays_from_bytes(payload)

    assert len(restored) == 2
    assert restored[0].shape == (2, 3)
    assert restored[1].tolist() == [5, 4, 3]
