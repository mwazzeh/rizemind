"""Model definition and data utilities for the torch_split example.

The model is split at a single cut layer:

    Input (flattened 28x28) → [ClientHead: Linear(784→H) + ReLU]
                             → activations (H,)
                             → [ServerTail: Linear(H→10)]
                             → logits → loss

Dataset: MNIST via Flower Datasets (ylecun/mnist).  Clients can use either IID
or Dirichlet non-IID partitions of the training set.  Images are flattened to a
784-dim vector so the architecture stays a simple MLP and the split-learning
contract (one activation tensor per step) is unchanged.

Label forwarding convention
---------------------------
During the forward round the client packs the batch labels as a second
ndarray alongside the activations:

    parameters.tensors[0]  - activation float32 array, shape (B, hidden_dim)
    parameters.tensors[1]  - label int64 array,         shape (B,)

``extract_activation_and_labels`` unpacks this on the server side.
"""

from __future__ import annotations

import io
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import Parameters, Scalar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

# ---------------------------------------------------------------------------
# Partitioner config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration for MNIST train partitioning in the torch_split example."""

    kind: str = "iid"
    seed: int = 42
    dirichlet_alpha: float = 0.5
    dirichlet_min_partition_size: int = 10
    dirichlet_self_balancing: bool = False


def _scalar_to_bool(value: Scalar) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def partition_config_from_run_config(
    run_config: Mapping[str, Scalar],
) -> PartitionConfig:
    """Build a partition config from Flower run config values."""
    kind = str(run_config.get("partitioner", "iid")).strip().lower()
    if kind not in {"iid", "dirichlet"}:
        raise ValueError(
            f"partitioner must be 'iid' or 'dirichlet', got {kind!r}"
        )

    return PartitionConfig(
        kind=kind,
        seed=int(run_config.get("partition-seed", 42)),
        dirichlet_alpha=float(run_config.get("dirichlet-alpha", 0.5)),
        dirichlet_min_partition_size=int(
            run_config.get("dirichlet-min-partition-size", 10)
        ),
        dirichlet_self_balancing=_scalar_to_bool(
            run_config.get("dirichlet-self-balancing", False)
        ),
    )


# ---------------------------------------------------------------------------
# Singleton FederatedDataset (cached across client_fn calls within a process)
# ---------------------------------------------------------------------------

_fds_cache: dict[tuple[int, PartitionConfig], FederatedDataset] = {}


def _make_train_partitioner(
    num_partitions: int,
    partition_config: PartitionConfig,
):
    if partition_config.kind == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    return DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=partition_config.dirichlet_alpha,
        min_partition_size=partition_config.dirichlet_min_partition_size,
        self_balancing=partition_config.dirichlet_self_balancing,
        shuffle=True,
        seed=partition_config.seed,
    )


def _get_fds(
    num_partitions: int,
    partition_config: PartitionConfig,
) -> FederatedDataset:
    cache_key = (num_partitions, partition_config)
    if cache_key not in _fds_cache:
        _fds_cache[cache_key] = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={
                "train": _make_train_partitioner(
                    num_partitions=num_partitions,
                    partition_config=partition_config,
                ),
            },
        )
    return _fds_cache[cache_key]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# MNIST: single channel 28x28, normalised to roughly [-1, 1].
# Flatten is applied inside the transform so that ClientHead can be a plain
# Linear without any CNN layers.
_MNIST_TRANSFORMS = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
    lambda t: t.view(-1),   # (1, 28, 28) → (784,)
])


def _apply_transforms(batch: dict) -> dict:
    batch["image"] = [_MNIST_TRANSFORMS(img) for img in batch["image"]]
    return batch


# ---------------------------------------------------------------------------
# Cached partition tensors
# ---------------------------------------------------------------------------


class TensorDictDataset(Dataset):
    """Dataset backed by tensors but exposing dict samples like Flower Datasets."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "label": self.labels[index]}


@dataclass
class CachedTrainPartition:
    """Materialized client partition with O(1) batch lookup across rounds.

    The transformed train split is stored as tensors once per process.  Each
    epoch uses a deterministic shuffle order derived from ``shuffle_seed`` and
    the epoch number, so clients can fetch a batch by index without rewinding a
    ``DataLoader`` from the start on every Flower round.
    """

    images: torch.Tensor
    labels: torch.Tensor
    batch_size: int
    shuffle_seed: int = 42
    _cached_epoch: int | None = field(default=None, init=False, repr=False)
    _cached_indices: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def num_batches(self) -> int:
        """Return the number of mini-batches in the cached train split."""
        return ceil(len(self.labels) / self.batch_size)

    def _indices_for_epoch(self, epoch: int) -> torch.Tensor:
        if self._cached_epoch != epoch or self._cached_indices is None:
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed + epoch)
            self._cached_indices = torch.randperm(len(self.labels), generator=generator)
            self._cached_epoch = epoch
        return self._cached_indices

    def get_batch(self, batch_idx: int, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the batch for ``(epoch, batch_idx)`` using cached tensors."""
        if not 0 <= batch_idx < self.num_batches:
            raise IndexError(
                f"batch_idx must be in [0, {self.num_batches}), got {batch_idx}"
            )

        indices = self._indices_for_epoch(epoch)
        start = batch_idx * self.batch_size
        stop = min(start + self.batch_size, len(self.labels))
        batch_indices = indices[start:stop]
        return self.images[batch_indices], self.labels[batch_indices]


_partition_cache: dict[
    tuple[int, int, int, float, PartitionConfig],
    tuple[CachedTrainPartition, TensorDictDataset],
] = {}
_train_batch_count_cache: dict[
    tuple[int, int, float, PartitionConfig],
    tuple[int, ...],
] = {}


def _split_partition(partition, val_ratio: float):
    """Split a local partition into train and validation sets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    return partition.train_test_split(test_size=val_ratio, seed=42)


def _materialize_split(split: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a transformed dataset split into tensors once per process."""
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for sample in split:
        images.append(sample["image"])
        labels.append(int(sample["label"]))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partition_config: PartitionConfig | None = None,
    val_ratio: float = 0.1,
) -> tuple[CachedTrainPartition, DataLoader]:
    """Load a partition and return cached train data + valloader.

    Args:
        partition_id: Zero-based index of this client's partition.
        num_partitions: Total number of client partitions.
        batch_size: Mini-batch size for both loaders.
        partition_config: Train partitioning configuration.
        val_ratio: Fraction of the partition reserved for validation.

    Returns:
        ``(train_partition, valloader)`` over the client's local data.
    """
    partition_config = partition_config or PartitionConfig()
    cache_key = (partition_id, num_partitions, batch_size, val_ratio, partition_config)
    if cache_key not in _partition_cache:
        fds = _get_fds(num_partitions, partition_config)
        partition = fds.load_partition(partition_id)
        splits = _split_partition(partition, val_ratio)
        splits = splits.with_transform(_apply_transforms)

        train_images, train_labels = _materialize_split(cast(Dataset, splits["train"]))
        val_images, val_labels = _materialize_split(cast(Dataset, splits["test"]))

        _partition_cache[cache_key] = (
            CachedTrainPartition(
                images=train_images,
                labels=train_labels,
                batch_size=batch_size,
                shuffle_seed=42 + partition_id,
            ),
            TensorDictDataset(val_images, val_labels),
        )

    train_partition, val_dataset = _partition_cache[cache_key]
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_partition, valloader


def get_train_batch_counts(
    num_partitions: int,
    batch_size: int,
    partition_config: PartitionConfig | None = None,
    val_ratio: float = 0.1,
) -> tuple[int, ...]:
    """Return train mini-batch counts for every client partition."""
    partition_config = partition_config or PartitionConfig()
    cache_key = (num_partitions, batch_size, val_ratio, partition_config)
    if cache_key not in _train_batch_count_cache:
        fds = _get_fds(num_partitions, partition_config)
        batch_counts = []
        for partition_id in range(num_partitions):
            partition = fds.load_partition(partition_id)
            train_split = _split_partition(partition, val_ratio)["train"]
            batch_counts.append(ceil(len(train_split) / batch_size))
        _train_batch_count_cache[cache_key] = tuple(batch_counts)
    return _train_batch_count_cache[cache_key]


def num_server_rounds_for_target_epochs(
    target_epochs: float,
    train_batch_counts: Sequence[int],
    num_rounds_per_step: int = 2,
) -> int:
    """Convert local target epochs into total Flower rounds.

    The largest train partition determines the number of split-learning steps
    needed to cover one full local epoch across all clients.  Smaller
    partitions may wrap and start a new epoch slightly earlier under non-IID
    partitioning.
    """
    if target_epochs <= 0.0:
        raise ValueError(f"target_epochs must be > 0, got {target_epochs}")
    if num_rounds_per_step < 1:
        raise ValueError(
            f"num_rounds_per_step must be >= 1, got {num_rounds_per_step}"
        )
    if not train_batch_counts:
        raise ValueError("train_batch_counts must not be empty")

    max_train_batches = max(train_batch_counts)
    target_steps = ceil(target_epochs * max_train_batches)
    return target_steps * num_rounds_per_step


def make_server_eval_loader(batch_size: int) -> DataLoader:
    """Return a DataLoader over the full MNIST test split for server evaluation.

    This is used by the server's ``evaluate_fn`` to compute test accuracy using
    the averaged client head weights and the current server tail.
    """
    from flwr_datasets import FederatedDataset

    test_fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"test": IidPartitioner(num_partitions=1)},
    )
    test_partition = test_fds.load_partition(0, split="test")
    test_partition = test_partition.with_transform(_apply_transforms)
    return DataLoader(cast(Dataset, test_partition), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ClientHead(nn.Module):
    """Client-side layers up to (and including) the cut layer.

    Accepts a flattened 784-dimensional MNIST input and produces a
    ``hidden_dim``-dimensional activation at the cut point.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class ServerTail(nn.Module):
    """Server-side layers after the cut layer."""

    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def get_weights(model: nn.Module) -> list:
    """Extract model weights as a list of numpy arrays."""
    return [val.detach().cpu().numpy().copy() for val in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list) -> None:
    """Load weights from a list of numpy arrays into a model."""
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)


def serialize_ndarrays_to_bytes(arrays: Sequence[np.ndarray]) -> bytes:
    """Serialize a list of ndarrays into a compact bytes payload."""
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, *arrays)
        return buffer.getvalue()


def deserialize_ndarrays_from_bytes(payload: bytes) -> list[np.ndarray]:
    """Deserialize ndarrays produced by ``serialize_ndarrays_to_bytes``."""
    def key_index(name: str) -> int:
        return int(name.split("_")[1])

    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return [data[key].copy() for key in sorted(data.files, key=key_index)]


# ---------------------------------------------------------------------------
# SL wire-format helpers
# ---------------------------------------------------------------------------


def extract_activation_and_labels(
    params: Parameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack activation and labels from a forward-round ``Parameters`` object.

    The client packs two arrays: ``tensors[0]`` is the activation (float32)
    and ``tensors[1]`` is the batch labels (int64).

    Args:
        params: ``Parameters`` produced during a split-learning forward round.

    Returns:
        ``(activation, labels)`` where ``activation`` is a leaf tensor with
        ``requires_grad=True`` and ``labels`` is a ``torch.int64`` tensor.
    """
    ndarrays = parameters_to_ndarrays(params)
    activation = torch.from_numpy(ndarrays[0]).requires_grad_(True)
    labels = torch.from_numpy(ndarrays[1].copy()).long()
    return activation, labels
