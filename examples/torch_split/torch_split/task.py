"""Model definitions and data utilities for the torch_split example.

Two datasets / architectures are supported, selected by ``dataset``:

- ``"mnist"`` — single-channel 28x28 images flattened to 784 dims, trained
  with a 2-layer MLP split at the hidden layer::

      Input (784,) → [ClientHead: Linear(784→H) + ReLU]
                    → activations (H,)
                    → [ServerTail: Linear(H→10)] → logits → loss

- ``"cifar10"`` — 3x32x32 images kept as CHW tensors, trained with a small
  CNN split after two conv+pool blocks::

      Input (3,32,32) → [ConvClientHead: Conv→ReLU→Pool→Conv→ReLU→Pool]
                       → activations (64,8,8)
                       → [ConvServerTail: Flatten→Linear→ReLU→Linear] → logits

In both cases the split-learning contract is unchanged: the client sends a
single activation tensor (plus the batch labels) to the server each forward
round.

Label forwarding convention
---------------------------
During the forward round the client packs the batch labels as a second
ndarray alongside the activations:

    parameters.tensors[0]  - activation float32 array, shape (B, *act_dims)
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
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a supported dataset and its architecture.

    Attributes:
        name: Short key (``"mnist"`` or ``"cifar10"``).
        hf_path: Hugging Face dataset path used by Flower Datasets.
        image_key: Column holding the image in the raw dataset.
        num_classes: Number of label classes.
        arch: ``"mlp"`` (flatten + linear) or ``"cnn"`` (keep CHW + conv).
        in_channels: Image channel count.
        image_hw: Image height/width (square images assumed).
        norm_mean: Per-channel normalization mean.
        norm_std: Per-channel normalization std.
    """

    name: str
    hf_path: str
    image_key: str
    num_classes: int
    arch: str
    in_channels: int
    image_hw: int
    norm_mean: tuple[float, ...]
    norm_std: tuple[float, ...]

    @property
    def input_dim(self) -> int:
        """Flattened input size (used by the MLP architecture)."""
        return self.in_channels * self.image_hw * self.image_hw


_DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        hf_path="ylecun/mnist",
        image_key="image",
        num_classes=10,
        arch="mlp",
        in_channels=1,
        image_hw=28,
        norm_mean=(0.1307,),
        norm_std=(0.3081,),
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        hf_path="uoft-cs/cifar10",
        image_key="img",
        num_classes=10,
        arch="cnn",
        in_channels=3,
        image_hw=32,
        norm_mean=(0.4914, 0.4822, 0.4465),
        norm_std=(0.2470, 0.2435, 0.2616),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name``.

    Args:
        name: Dataset key, case-insensitive (``"mnist"`` or ``"cifar10"``).

    Returns:
        The matching dataset specification.

    Raises:
        ValueError: If ``name`` is not a supported dataset.
    """
    key = name.strip().lower()
    if key not in _DATASETS:
        raise ValueError(f"dataset must be one of {sorted(_DATASETS)}, got {name!r}")
    return _DATASETS[key]


# ---------------------------------------------------------------------------
# Partitioner config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration for train partitioning in the torch_split example."""

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
        raise ValueError(f"partitioner must be 'iid' or 'dirichlet', got {kind!r}")

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

_fds_cache: dict[tuple[str, int, PartitionConfig], FederatedDataset] = {}


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
    spec: DatasetSpec,
    num_partitions: int,
    partition_config: PartitionConfig,
) -> FederatedDataset:
    cache_key = (spec.name, num_partitions, partition_config)
    if cache_key not in _fds_cache:
        _fds_cache[cache_key] = FederatedDataset(
            dataset=spec.hf_path,
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


def _build_transform(spec: DatasetSpec):
    """Return the image transform for ``spec``.

    MLP datasets flatten to a 1-D vector; CNN datasets keep CHW layout.
    """
    steps: list = [ToTensor(), Normalize(spec.norm_mean, spec.norm_std)]
    if spec.arch == "mlp":
        steps.append(lambda t: t.view(-1))
    return Compose(steps)


def _make_apply_transforms(spec: DatasetSpec):
    """Return a batch transform that writes the tensor under ``"image"``.

    The raw source column is dropped when it is not already ``"image"`` so a
    plain ``DataLoader`` over the transformed split does not try to collate the
    original PIL images.
    """
    transform = _build_transform(spec)

    def _apply(batch: dict) -> dict:
        images = [transform(img) for img in batch[spec.image_key]]
        if spec.image_key != "image":
            del batch[spec.image_key]
        batch["image"] = images
        return batch

    return _apply


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

    def get_batch(
        self, batch_idx: int, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    tuple[str, int, int, int, float, int, PartitionConfig],
    tuple[CachedTrainPartition, TensorDictDataset],
] = {}
_train_batch_count_cache: dict[
    tuple[str, int, int, float, int, PartitionConfig],
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


def _capped_size(n: int, max_train_samples: int | None) -> int:
    """Return ``min(n, max_train_samples)`` treating non-positive caps as off."""
    if max_train_samples is None or max_train_samples <= 0:
        return n
    return min(n, max_train_samples)


def _subsample(
    images: torch.Tensor,
    labels: torch.Tensor,
    max_train_samples: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take a deterministic random subset to keep epochs cheap.

    A seeded permutation avoids label skew from any ordering in the source
    split. Returns the inputs unchanged when no cap applies.
    """
    cap = _capped_size(len(labels), max_train_samples)
    if cap == len(labels):
        return images, labels
    generator = torch.Generator()
    generator.manual_seed(seed)
    keep = torch.randperm(len(labels), generator=generator)[:cap]
    return images[keep], labels[keep]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    spec: DatasetSpec,
    partition_config: PartitionConfig | None = None,
    val_ratio: float = 0.1,
    max_train_samples: int | None = None,
) -> tuple[CachedTrainPartition, DataLoader]:
    """Load a partition and return cached train data + valloader.

    Args:
        partition_id: Zero-based index of this client's partition.
        num_partitions: Total number of client partitions.
        batch_size: Mini-batch size for both loaders.
        spec: Dataset specification.
        partition_config: Train partitioning configuration.
        val_ratio: Fraction of the partition reserved for validation.
        max_train_samples: Optional per-client cap on training samples
            (keeps full-epoch sweeps cheap). ``None`` or ``<= 0`` disables it.

    Returns:
        ``(train_partition, valloader)`` over the client's local data.
    """
    partition_config = partition_config or PartitionConfig()
    cache_key = (
        spec.name,
        partition_id,
        num_partitions,
        batch_size,
        val_ratio,
        max_train_samples or 0,
        partition_config,
    )
    if cache_key not in _partition_cache:
        fds = _get_fds(spec, num_partitions, partition_config)
        partition = fds.load_partition(partition_id)
        splits = _split_partition(partition, val_ratio)
        splits = splits.with_transform(_make_apply_transforms(spec))

        train_images, train_labels = _materialize_split(cast(Dataset, splits["train"]))
        train_images, train_labels = _subsample(
            train_images, train_labels, max_train_samples, seed=1234 + partition_id
        )
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
    spec: DatasetSpec,
    partition_config: PartitionConfig | None = None,
    val_ratio: float = 0.1,
    max_train_samples: int | None = None,
) -> tuple[int, ...]:
    """Return train mini-batch counts for every client partition."""
    partition_config = partition_config or PartitionConfig()
    cache_key = (
        spec.name,
        num_partitions,
        batch_size,
        val_ratio,
        max_train_samples or 0,
        partition_config,
    )
    if cache_key not in _train_batch_count_cache:
        fds = _get_fds(spec, num_partitions, partition_config)
        batch_counts = []
        for partition_id in range(num_partitions):
            partition = fds.load_partition(partition_id)
            train_split = _split_partition(partition, val_ratio)["train"]
            n_train = _capped_size(len(train_split), max_train_samples)
            batch_counts.append(ceil(n_train / batch_size))
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
        raise ValueError(f"num_rounds_per_step must be >= 1, got {num_rounds_per_step}")
    if not train_batch_counts:
        raise ValueError("train_batch_counts must not be empty")

    max_train_batches = max(train_batch_counts)
    target_steps = ceil(target_epochs * max_train_batches)
    return target_steps * num_rounds_per_step


def make_server_eval_loader(
    batch_size: int,
    spec: DatasetSpec,
    max_samples: int | None = None,
) -> DataLoader:
    """Return a DataLoader over the test split for server evaluation.

    This is used by the server's ``evaluate_fn`` to compute test accuracy using
    the averaged client head weights and the current server tail.

    Args:
        batch_size: Evaluation batch size.
        spec: Dataset specification.
        max_samples: Optional cap on the number of test images (a held-out
            subset, useful to keep frequent evaluation cheap). ``None`` or
            ``<= 0`` uses the full test split.

    Returns:
        A ``DataLoader`` over the (optionally capped) test split.
    """
    test_fds = FederatedDataset(
        dataset=spec.hf_path,
        partitioners={"test": IidPartitioner(num_partitions=1)},
    )
    test_partition = test_fds.load_partition(0, split="test")
    if max_samples and max_samples > 0 and len(test_partition) > max_samples:
        test_partition = test_partition.select(range(max_samples))
    test_partition = test_partition.with_transform(_make_apply_transforms(spec))
    return DataLoader(
        cast(Dataset, test_partition), batch_size=batch_size, shuffle=False
    )


# ---------------------------------------------------------------------------
# Models — MNIST MLP
# ---------------------------------------------------------------------------


class ClientHead(nn.Module):
    """Client-side MLP layers up to (and including) the cut layer.

    Accepts a flattened input and produces a ``hidden_dim``-dimensional
    activation at the cut point.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class ServerTail(nn.Module):
    """Server-side MLP layers after the cut layer."""

    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Models — CIFAR-10 CNN
# ---------------------------------------------------------------------------


class ConvClientHead(nn.Module):
    """Client-side CNN: two conv+ReLU+pool blocks.

    For a 3x32x32 input the activation at the cut point has shape
    ``(64, 8, 8)``.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x


class ConvServerTail(nn.Module):
    """Server-side CNN classifier head.

    Flattens the ``(64, H, W)`` activation and runs two linear layers.
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 64 * 8 * 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def build_split_models(
    spec: DatasetSpec,
    hidden_dim: int = 128,
    num_classes: int | None = None,
) -> tuple[nn.Module, nn.Module]:
    """Construct the (head, tail) pair for a dataset's architecture.

    Args:
        spec: Dataset specification selecting the architecture.
        hidden_dim: Hidden width (MLP cut size / CNN tail width).
        num_classes: Override for the number of output classes; defaults to
            ``spec.num_classes``.

    Returns:
        ``(head, tail)`` modules. The head runs on the client, the tail on the
        server.
    """
    classes = spec.num_classes if num_classes is None else num_classes
    if spec.arch == "mlp":
        head: nn.Module = ClientHead(spec.input_dim, hidden_dim)
        tail: nn.Module = ServerTail(hidden_dim, classes)
        return head, tail

    feature_hw = spec.image_hw // 4  # two stride-2 pools
    feature_dim = 64 * feature_hw * feature_hw
    head = ConvClientHead(spec.in_channels)
    tail = ConvServerTail(classes, feature_dim=feature_dim, hidden_dim=hidden_dim)
    return head, tail


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
