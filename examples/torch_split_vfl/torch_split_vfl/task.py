"""Model + data utilities for the torch_split_vfl example.

This example demonstrates **vertical** split federated learning (VFL-SFL):
every client holds the *same* MNIST samples but a *different* vertical strip
of each 28x28 image. The server holds the labels and a small MLP tail that
classifies the concatenated bottom-model activations.

Per-step sample alignment
-------------------------
Every party (each client + the server) constructs the *same* deterministic
shuffle order over the MNIST train split. Given an ``sl_step``, all parties
derive the same ``(epoch, batch_idx)`` and pull the same sample ids — clients
return their feature-slice activations and the server pulls the matching
labels. The shared shuffle seed is :data:`VFL_SHUFFLE_SEED`.

Strip layout
------------
For ``num_clients = K`` and ``image_hw = W`` (28 for MNIST), each client
``partition_id = p`` owns columns ``[p*W//K, (p+1)*W//K)`` of every image (the
last client absorbs the remainder when ``W`` is not divisible by ``K``).
Bottom models are MLPs: ``Linear(strip_features → hidden) → ReLU``.

Server tail
-----------
The tail accepts the concatenation of ``K`` activations of size
``hidden_dim`` each, i.e. an input of size ``K * hidden_dim``, and outputs
``num_classes`` logits.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import Parameters
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

#: Permutation seed shared by clients + server to keep batches aligned.
VFL_SHUFFLE_SEED = 42

# ---------------------------------------------------------------------------
# Dataset registry (only MNIST in v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularSpec:
    """Vertical split description for a tabular dataset.

    Each entry of ``party_columns`` lists the raw columns owned by one party, so
    ``len(party_columns)`` is the number of vertical participants (K). Columns
    are looked up in ``numerical`` / ``categorical`` to decide encoding; the
    label column is held by the server and never handed to a party.

    Attributes:
        hf_path: Hugging Face dataset path (loaded via ``datasets``).
        label_column: Column holding the classification target.
        positive_label: Value of ``label_column`` mapped to class 1.
        numerical: Columns standardized to zero mean / unit variance (train
            statistics).
        categorical: Columns one-hot encoded against the train vocabulary.
        party_columns: One tuple of column names per party (the vertical split).
        test_fraction: Fraction of rows held out as the test split.
        split_seed: Seed for the deterministic train/test split.
    """

    hf_path: str
    label_column: str
    positive_label: str
    numerical: tuple[str, ...]
    categorical: tuple[str, ...]
    party_columns: tuple[tuple[str, ...], ...]
    test_fraction: float = 0.2
    split_seed: int = 12345


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a supported dataset for vertical split learning.

    Two kinds are supported. For ``kind == "image"`` the image-* fields describe
    the data and parties own vertical column strips. For ``kind == "tabular"``
    the ``tabular`` field describes the column split and the image-* fields are
    unused.

    Attributes:
        name: Short key (``"mnist"``, ``"adult"``).
        kind: ``"image"`` or ``"tabular"``.
        num_classes: Number of label classes.
        hf_path: Hugging Face dataset path (image datasets).
        image_key: Column holding the image in the raw dataset.
        in_channels: Image channel count.
        image_hw: Image height/width (square images assumed).
        norm_mean: Per-channel normalization mean.
        norm_std: Per-channel normalization std.
        tabular: Tabular split spec (tabular datasets).
    """

    name: str
    kind: str
    num_classes: int
    hf_path: str | None = None
    image_key: str | None = None
    in_channels: int | None = None
    image_hw: int | None = None
    norm_mean: tuple[float, ...] | None = None
    norm_std: tuple[float, ...] | None = None
    tabular: TabularSpec | None = None


_DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        kind="image",
        num_classes=10,
        hf_path="ylecun/mnist",
        image_key="image",
        in_channels=1,
        image_hw=28,
        norm_mean=(0.1307,),
        norm_std=(0.3081,),
    ),
    # UCI Adult census income — a realistic tabular VFL case where parties own
    # semantically different columns (demographics vs. work/financial). Party 0
    # holds person/demographic columns, party 1 holds work/education/financial
    # columns, and the server holds the income label.
    "adult": DatasetSpec(
        name="adult",
        kind="tabular",
        num_classes=2,
        tabular=TabularSpec(
            hf_path="scikit-learn/adult-census-income",
            label_column="income",
            positive_label=">50K",
            numerical=(
                "age",
                "fnlwgt",
                "education.num",
                "capital.gain",
                "capital.loss",
                "hours.per.week",
            ),
            categorical=(
                "workclass",
                "education",
                "marital.status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native.country",
            ),
            party_columns=(
                # Party 0 — demographic / person.
                ("age", "sex", "race", "marital.status", "relationship",
                 "native.country"),
                # Party 1 — work / education / financial.
                ("workclass", "education", "education.num", "occupation",
                 "hours.per.week", "capital.gain", "capital.loss", "fnlwgt"),
            ),
        ),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name``.

    Args:
        name: Dataset key (case-insensitive): ``"mnist"`` or ``"adult"``.

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
# Strip layout
# ---------------------------------------------------------------------------


def strip_bounds(partition_id: int, num_clients: int, image_hw: int) -> tuple[int, int]:
    """Return the ``[start_col, stop_col)`` strip range for ``partition_id``.

    The image width is split into ``num_clients`` strips; if ``image_hw`` is
    not divisible by ``num_clients`` the last strip absorbs the remainder.

    Args:
        partition_id: Zero-based client id.
        num_clients: Total number of vertical participants.
        image_hw: Image width (and height; square images).

    Returns:
        ``(start_col, stop_col)`` such that the client owns columns
        ``[start_col, stop_col)``.

    Raises:
        ValueError: If ``partition_id`` is outside ``[0, num_clients)`` or
            ``image_hw < num_clients``.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if not 0 <= partition_id < num_clients:
        raise ValueError(
            f"partition_id must be in [0, {num_clients}), got {partition_id}"
        )
    if image_hw < num_clients:
        raise ValueError(f"image_hw={image_hw} must be >= num_clients={num_clients}")
    base = image_hw // num_clients
    start = partition_id * base
    stop = (partition_id + 1) * base if partition_id < num_clients - 1 else image_hw
    return start, stop


def strip_feature_dim(spec: DatasetSpec, partition_id: int, num_clients: int) -> int:
    """Return the flattened feature size of one client's strip."""
    start, stop = strip_bounds(partition_id, num_clients, spec.image_hw)
    return spec.in_channels * spec.image_hw * (stop - start)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _build_transform(spec: DatasetSpec):
    """Image transform: ToTensor + Normalize, keeping CHW for column slicing."""
    return Compose([ToTensor(), Normalize(spec.norm_mean, spec.norm_std)])


def _apply_transform(spec: DatasetSpec):
    transform = _build_transform(spec)

    def _apply(batch: dict) -> dict:
        images = [transform(img) for img in batch[spec.image_key]]
        if spec.image_key != "image":
            del batch[spec.image_key]
        batch["image"] = images
        return batch

    return _apply


def _materialize(split) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for sample in split:
        images.append(sample["image"])
        labels.append(int(sample["label"]))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


_full_train_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_full_test_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def load_full_train(spec: DatasetSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, labels)`` for the full train split, cached per process.

    Args:
        spec: Dataset specification.

    Returns:
        Tuple of ``(images, labels)`` tensors of shape ``(N, C, H, W)`` and
        ``(N,)`` respectively.
    """
    if spec.name not in _full_train_cache:
        fds = FederatedDataset(
            dataset=spec.hf_path,
            partitioners={"train": IidPartitioner(num_partitions=1)},
        )
        part = fds.load_partition(0, split="train")
        part = part.with_transform(_apply_transform(spec))
        _full_train_cache[spec.name] = _materialize(cast(Dataset, part))
    return _full_train_cache[spec.name]


def load_full_test(spec: DatasetSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, labels)`` for the full test split, cached per process."""
    if spec.name not in _full_test_cache:
        fds = FederatedDataset(
            dataset=spec.hf_path,
            partitioners={"test": IidPartitioner(num_partitions=1)},
        )
        part = fds.load_partition(0, split="test")
        part = part.with_transform(_apply_transform(spec))
        _full_test_cache[spec.name] = _materialize(cast(Dataset, part))
    return _full_test_cache[spec.name]


# ---------------------------------------------------------------------------
# Tabular preprocessing (deterministic, leakage-free, sample-aligned)
# ---------------------------------------------------------------------------


Table = dict[str, list]


@dataclass(frozen=True)
class PreprocessedTabular:
    """Fully numeric, party-split view of a tabular dataset.

    Row ``i`` refers to the same sample across every party tensor and the label
    tensors, so the shared-permutation batching in :class:`VerticalPartition`
    keeps all parties aligned.

    Attributes:
        party_train: One ``(N_train, d_p)`` float tensor per party.
        party_test: One ``(N_test, d_p)`` float tensor per party.
        train_labels: ``(N_train,)`` long tensor of class ids.
        test_labels: ``(N_test,)`` long tensor of class ids.
        party_dims: Encoded feature width ``d_p`` per party.
    """

    party_train: list[torch.Tensor]
    party_test: list[torch.Tensor]
    train_labels: torch.Tensor
    test_labels: torch.Tensor
    party_dims: list[int]


def validate_tabular_spec(tspec: TabularSpec) -> None:
    """Check that the column split is well-formed and label-safe.

    Raises:
        ValueError: If a party column is unknown, if the label column leaks into
            a party, or if a column is assigned to more than one party.
    """
    known = set(tspec.numerical) | set(tspec.categorical)
    seen: set[str] = set()
    for pid, cols in enumerate(tspec.party_columns):
        for col in cols:
            if col == tspec.label_column:
                raise ValueError(
                    f"label column {tspec.label_column!r} must not be assigned "
                    f"to a party (found in party {pid})"
                )
            if col not in known:
                raise ValueError(
                    f"party {pid} column {col!r} is neither numerical nor "
                    "categorical"
                )
            if col in seen:
                raise ValueError(f"column {col!r} is assigned to multiple parties")
            seen.add(col)


def deterministic_split(
    table: Table, test_fraction: float, seed: int
) -> tuple[Table, Table]:
    """Split a column table into ``(train, test)`` with a fixed permutation.

    The same ``seed`` yields the same split on every party/process, so all
    parties agree on which rows are train vs. test and in what order.
    """
    n = len(next(iter(table.values())))
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_test = int(round(n * test_fraction))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def take(idx: list[int]) -> Table:
        return {col: [table[col][i] for i in idx] for col in table}

    return take(train_idx), take(test_idx)


def _clean(value: object) -> str:
    """Normalize a raw categorical/label cell to a stripped string."""
    return str(value).strip()


def preprocess_tabular(
    train_table: Table, test_table: Table, tspec: TabularSpec
) -> PreprocessedTabular:
    """Encode a tabular dataset into per-party numeric tensors.

    All encoders are fit on the **train** split only (no test leakage):
    numerical columns are standardized with train mean/std; categorical columns
    are one-hot encoded against the sorted train vocabulary, with unseen test
    categories mapped to an all-zero vector. Missing values (e.g. ``"?"``) are
    treated as an ordinary category. Row order is preserved, so the parties and
    labels stay aligned.

    Args:
        train_table: Column-oriented train rows (``{column: [values]}``).
        test_table: Column-oriented test rows.
        tspec: Tabular split specification.

    Returns:
        A :class:`PreprocessedTabular` with per-party tensors and labels.
    """
    validate_tabular_spec(tspec)

    num_stats: dict[str, tuple[float, float]] = {}
    for col in tspec.numerical:
        vals = [float(v) for v in train_table[col]]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        num_stats[col] = (mean, std if std > 0 else 1.0)

    cat_vocab: dict[str, dict[str, int]] = {}
    for col in tspec.categorical:
        uniq = sorted({_clean(v) for v in train_table[col]})
        cat_vocab[col] = {val: i for i, val in enumerate(uniq)}

    def encode_column(table: Table, col: str) -> torch.Tensor:
        if col in num_stats:
            mean, std = num_stats[col]
            return torch.tensor(
                [[(float(v) - mean) / std] for v in table[col]], dtype=torch.float32
            )
        vocab = cat_vocab[col]
        out = torch.zeros((len(table[col]), len(vocab)), dtype=torch.float32)
        for i, v in enumerate(table[col]):
            idx = vocab.get(_clean(v))
            if idx is not None:  # unseen test category -> all-zero row
                out[i, idx] = 1.0
        return out

    def build_party(table: Table, cols: tuple[str, ...]) -> torch.Tensor:
        return torch.cat([encode_column(table, c) for c in cols], dim=1)

    def encode_labels(table: Table) -> torch.Tensor:
        pos = tspec.positive_label
        ys = [1 if _clean(v).rstrip(".") == pos else 0 for v in table[tspec.label_column]]
        return torch.tensor(ys, dtype=torch.long)

    party_train = [build_party(train_table, cols) for cols in tspec.party_columns]
    party_test = [build_party(test_table, cols) for cols in tspec.party_columns]
    return PreprocessedTabular(
        party_train=party_train,
        party_test=party_test,
        train_labels=encode_labels(train_table),
        test_labels=encode_labels(test_table),
        party_dims=[t.shape[1] for t in party_train],
    )


_tabular_cache: dict[str, PreprocessedTabular] = {}


def load_tabular(spec: DatasetSpec) -> PreprocessedTabular:
    """Return the cached :class:`PreprocessedTabular` for a tabular ``spec``.

    Downloads the dataset, applies the deterministic train/test split, and
    encodes it. Cached per process and per dataset name.

    Raises:
        ValueError: If ``spec`` is not a tabular dataset.
    """
    if spec.kind != "tabular" or spec.tabular is None:
        raise ValueError(f"{spec.name!r} is not a tabular dataset")
    if spec.name not in _tabular_cache:
        from datasets import load_dataset

        tspec = spec.tabular
        ds = load_dataset(tspec.hf_path)["train"]
        table: Table = {col: list(ds[col]) for col in ds.column_names}
        train_table, test_table = deterministic_split(
            table, tspec.test_fraction, tspec.split_seed
        )
        _tabular_cache[spec.name] = preprocess_tabular(train_table, test_table, tspec)
    return _tabular_cache[spec.name]


def _validate_tabular_clients(spec: DatasetSpec, num_clients: int) -> None:
    """Ensure ``num_clients`` matches the number of tabular parties."""
    assert spec.tabular is not None
    expected = len(spec.tabular.party_columns)
    if num_clients != expected:
        raise ValueError(
            f"dataset {spec.name!r} defines {expected} vertical parties; "
            f"set min-available-clients={expected} (got {num_clients})"
        )


def parse_active_parties(value: object) -> tuple[int, ...] | None:
    """Parse the ``active-parties`` run-config value into group ids.

    Accepts a comma-separated string (e.g. ``"0,1"``, ``"1"``), an empty string
    or ``"all"`` (meaning "all configured groups", returned as ``None``), or
    ``None``.

    Args:
        value: Raw config value.

    Returns:
        Tuple of group ids, or ``None`` for the default (all groups).

    Raises:
        ValueError: If a token is not an integer.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "all"):
        return None
    try:
        return tuple(int(tok) for tok in s.replace(" ", "").split(",") if tok != "")
    except ValueError as exc:
        raise ValueError(
            f"active-parties must be a comma-separated list of integers, got {value!r}"
        ) from exc


def num_configured_parties(spec: DatasetSpec, num_clients: int) -> int:
    """Return the number of feature groups the dataset defines.

    For tabular datasets this is the number of column groups in the spec; for
    image datasets each of the ``num_clients`` vertical strips is a party.
    """
    if spec.kind == "tabular":
        assert spec.tabular is not None
        return len(spec.tabular.party_columns)
    return num_clients


def resolve_active_groups(
    spec: DatasetSpec,
    num_clients: int,
    active_parties: Sequence[int] | None = None,
) -> list[int]:
    """Map active client slots to dataset feature-group ids.

    The returned list has length ``num_clients``; entry ``i`` is the feature
    group that client ``partition_id=i`` owns. This is what makes genuine
    federated feature ablations possible: e.g. a 1-client run with
    ``active_parties=[1]`` trains only on Adult's work/financial group.

    Args:
        spec: Dataset specification.
        num_clients: Number of active vertical participants (K).
        active_parties: Optional explicit feature-group ids to activate, in
            client order. ``None`` means "all configured groups in order".

    Returns:
        Feature-group id per client slot.

    Raises:
        ValueError: For tabular datasets, if the count, range, or uniqueness of
            ``active_parties`` is invalid. For image datasets, if a non-trivial
            subset is requested (strip subsetting is not supported).
    """
    if spec.kind == "tabular":
        assert spec.tabular is not None
        n_groups = len(spec.tabular.party_columns)
        groups = list(range(n_groups)) if active_parties is None else list(active_parties)
        if len(groups) == 0:
            raise ValueError("active_parties must select at least one feature group")
        if len(groups) != num_clients:
            raise ValueError(
                f"active_parties selects {len(groups)} group(s) {groups} but "
                f"min-available-clients={num_clients}; they must match"
            )
        for g in groups:
            if not 0 <= g < n_groups:
                raise ValueError(
                    f"active party {g} out of range; dataset {spec.name!r} has "
                    f"{n_groups} feature groups (valid ids 0..{n_groups - 1})"
                )
        if len(set(groups)) != len(groups):
            raise ValueError(f"active_parties has duplicate group ids: {groups}")
        return groups

    # Image datasets: parties are vertical strips indexed by partition_id.
    if active_parties is not None and list(active_parties) != list(range(num_clients)):
        raise ValueError(
            "active-parties subset selection is only supported for tabular "
            "datasets; image strips are defined by num_clients"
        )
    return list(range(num_clients))


# ---------------------------------------------------------------------------
# Vertical partitions (shared sample order across all parties)
# ---------------------------------------------------------------------------


@dataclass
class VerticalPartition:
    """Per-party view of the train split.

    Holds *all* training samples in the party's local feature shape (a strip
    of columns for a client, or the full label tensor for the server). Index
    sequencing is shared: given an ``sl_step``, both clients and the server
    derive the same ``(epoch, batch_idx)`` and pull the same sample ids using
    a permutation seeded by :data:`VFL_SHUFFLE_SEED`.

    Attributes:
        features: Tensor of shape ``(N, ...)`` — one row per training sample.
            For clients this is the feature strip; for the server-as-labels
            it is the label tensor passed as a 1-D feature column.
        batch_size: Mini-batch size shared by all parties.
        num_total: Effective number of training samples (after optional cap).
    """

    features: torch.Tensor
    batch_size: int
    shuffle_seed: int = VFL_SHUFFLE_SEED
    _cached_epoch: int | None = field(default=None, init=False, repr=False)
    _cached_perm: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def num_total(self) -> int:
        return len(self.features)

    @property
    def num_batches(self) -> int:
        return ceil(self.num_total / self.batch_size)

    def step_to_epoch_batch(self, sl_step: int) -> tuple[int, int]:
        """Map a global step counter to ``(epoch, batch_idx)``."""
        if sl_step < 0:
            raise ValueError(f"sl_step must be >= 0, got {sl_step}")
        epoch, batch_idx = divmod(sl_step, self.num_batches)
        return epoch, batch_idx

    def _permutation_for_epoch(self, epoch: int) -> torch.Tensor:
        if self._cached_epoch != epoch or self._cached_perm is None:
            gen = torch.Generator()
            gen.manual_seed(self.shuffle_seed + epoch)
            self._cached_perm = torch.randperm(self.num_total, generator=gen)
            self._cached_epoch = epoch
        return self._cached_perm

    def get_batch_for_step(self, sl_step: int) -> torch.Tensor:
        """Return the batch slice for ``sl_step``.

        Args:
            sl_step: Global step counter (incremented after each backward
                round on the server).

        Returns:
            Tensor of shape ``(B, ...)`` where ``B`` may be smaller than
            ``batch_size`` on the last partial batch.
        """
        epoch, batch_idx = self.step_to_epoch_batch(sl_step)
        perm = self._permutation_for_epoch(epoch)
        start = batch_idx * self.batch_size
        stop = min(start + self.batch_size, self.num_total)
        return self.features[perm[start:stop]]


def _maybe_cap(tensor: torch.Tensor, max_samples: int | None) -> torch.Tensor:
    """Take the first ``max_samples`` rows; ``None``/``<= 0`` keeps everything."""
    if max_samples is None or max_samples <= 0:
        return tensor
    return tensor[: min(len(tensor), max_samples)]


def make_client_train_partition(
    spec: DatasetSpec,
    partition_id: int,
    num_clients: int,
    batch_size: int,
    max_train_samples: int | None = None,
    active_parties: Sequence[int] | None = None,
    shuffle_seed: int = VFL_SHUFFLE_SEED,
) -> VerticalPartition:
    """Build the client's vertical strip view of the train split.

    Args:
        spec: Dataset specification.
        partition_id: Zero-based client id.
        num_clients: Total number of vertical participants.
        batch_size: Mini-batch size.
        max_train_samples: Optional cap on training samples (``None`` / ``<= 0``
            uses the full train split).
        active_parties: Optional feature-group selection (tabular only); see
            :func:`resolve_active_groups`.
        shuffle_seed: Per-epoch batch-permutation seed; must be identical on
            every party so sample ids stay aligned.

    Returns:
        A :class:`VerticalPartition` whose feature dimension equals the
        client's strip (image) or column group (tabular), flattened.
    """
    if spec.kind == "tabular":
        group = resolve_active_groups(spec, num_clients, active_parties)[partition_id]
        features = load_tabular(spec).party_train[group]
    else:
        images, _labels = load_full_train(spec)
        start, stop = strip_bounds(partition_id, num_clients, spec.image_hw)
        strip = images[:, :, :, start:stop].contiguous()  # (N, C, H, strip_w)
        features = strip.view(strip.shape[0], -1)  # (N, C*H*strip_w)
    features = _maybe_cap(features, max_train_samples)
    return VerticalPartition(
        features=features, batch_size=batch_size, shuffle_seed=shuffle_seed
    )


def make_server_train_labels(
    spec: DatasetSpec,
    batch_size: int,
    max_train_samples: int | None = None,
    shuffle_seed: int = VFL_SHUFFLE_SEED,
) -> VerticalPartition:
    """Build the server's label-only view of the train split.

    Args:
        spec: Dataset specification.
        batch_size: Mini-batch size (must match clients).
        max_train_samples: Optional cap (must match the clients' cap).
        shuffle_seed: Must match the clients' ``shuffle_seed`` so the label
            batch lines up with the activation batch.

    Returns:
        A :class:`VerticalPartition` whose ``features`` tensor is the
        ``(N,)`` label vector — same sample order as the clients' strips.
    """
    if spec.kind == "tabular":
        labels = load_tabular(spec).train_labels
    else:
        _images, labels = load_full_train(spec)
    labels = _maybe_cap(labels, max_train_samples)
    return VerticalPartition(
        features=labels, batch_size=batch_size, shuffle_seed=shuffle_seed
    )


def party_feature_dim(
    spec: DatasetSpec,
    partition_id: int,
    num_clients: int,
    active_parties: Sequence[int] | None = None,
) -> int:
    """Return the encoded input width of one party's bottom model.

    Args:
        spec: Dataset specification.
        partition_id: Zero-based party id.
        num_clients: Total number of vertical participants.
        active_parties: Optional feature-group selection (tabular only).

    Returns:
        Flattened feature size for the party (image strip or tabular columns).
    """
    if spec.kind == "tabular":
        group = resolve_active_groups(spec, num_clients, active_parties)[partition_id]
        return load_tabular(spec).party_dims[group]
    return strip_feature_dim(spec, partition_id, num_clients)


def party_test_features(
    spec: DatasetSpec,
    partition_id: int,
    num_clients: int,
    max_samples: int | None = None,
    active_parties: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return one party's flattened test features ``(M, d_p)``.

    Used by the server's centralized evaluation. Row order matches
    :func:`test_labels`, so labels and features stay aligned.
    """
    if spec.kind == "tabular":
        group = resolve_active_groups(spec, num_clients, active_parties)[partition_id]
        return _maybe_cap(load_tabular(spec).party_test[group], max_samples)
    images, _labels = load_full_test(spec)
    images = _maybe_cap(images, max_samples)
    start, stop = strip_bounds(partition_id, num_clients, spec.image_hw)
    view = images[:, :, :, start:stop].contiguous()
    return view.view(view.shape[0], -1)


def test_labels(spec: DatasetSpec, max_samples: int | None = None) -> torch.Tensor:
    """Return the ``(M,)`` test label tensor, capped to ``max_samples`` rows."""
    if spec.kind == "tabular":
        return _maybe_cap(load_tabular(spec).test_labels, max_samples)
    _images, labels = load_full_test(spec)
    return _maybe_cap(labels, max_samples)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BottomMLP(nn.Module):
    """Client-side MLP that maps one feature strip to a ``hidden_dim`` vector.

    Args:
        in_features: Flattened strip size (``C * H * strip_w``).
        hidden_dim: Output activation dimension.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class ServerTopMLP(nn.Module):
    """Server-side classifier head on the concatenation of K bottom activations.

    Args:
        num_clients: Number of vertical participants.
        hidden_dim: Per-client activation size at the cut point.
        num_classes: Output class count.
    """

    def __init__(
        self, num_clients: int, hidden_dim: int = 64, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_clients * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def build_bottom_model(
    spec: DatasetSpec,
    partition_id: int,
    num_clients: int,
    hidden_dim: int = 64,
    active_parties: Sequence[int] | None = None,
) -> nn.Module:
    """Construct the bottom MLP for one client given its feature size."""
    in_features = party_feature_dim(spec, partition_id, num_clients, active_parties)
    return BottomMLP(in_features=in_features, hidden_dim=hidden_dim)


def build_server_top(
    spec: DatasetSpec, num_clients: int, hidden_dim: int = 64
) -> nn.Module:
    """Construct the server-side classifier head."""
    return ServerTopMLP(
        num_clients=num_clients, hidden_dim=hidden_dim, num_classes=spec.num_classes
    )


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Extract a model's parameters as a list of CPU numpy arrays."""
    return [val.detach().cpu().numpy().copy() for val in model.state_dict().values()]


def set_weights(model: nn.Module, weights: Sequence[np.ndarray]) -> None:
    """Load a list of numpy arrays back into a model in declaration order."""
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# SL wire-format helpers
# ---------------------------------------------------------------------------


def extract_activation(params: Parameters) -> torch.Tensor:
    """Unpack a client's activation ndarray into a leaf tensor with grad.

    Args:
        params: ``Parameters`` produced by the client in a forward round.

    Returns:
        Tensor with ``requires_grad=True`` ready for the server tail.

    Raises:
        ValueError: If ``params`` does not carry exactly one ndarray.
    """
    arrays = parameters_to_ndarrays(params)
    if len(arrays) != 1:
        raise ValueError(
            f"vertical activation must carry exactly 1 ndarray, got {len(arrays)}"
        )
    return torch.from_numpy(arrays[0]).requires_grad_(True)


# ---------------------------------------------------------------------------
# Round budget
# ---------------------------------------------------------------------------


def num_server_rounds_for_target_epochs(
    target_epochs: float,
    num_batches: int,
    num_rounds_per_step: int = 2,
) -> int:
    """Convert a target epoch count into the total Flower round budget.

    Args:
        target_epochs: Desired epochs over the train data.
        num_batches: Mini-batches per epoch.
        num_rounds_per_step: Flower rounds per VFL training step (default 2).

    Returns:
        Total Flower round budget.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if target_epochs <= 0.0:
        raise ValueError(f"target_epochs must be > 0, got {target_epochs}")
    if num_batches <= 0:
        raise ValueError(f"num_batches must be > 0, got {num_batches}")
    if num_rounds_per_step < 1:
        raise ValueError(f"num_rounds_per_step must be >= 1, got {num_rounds_per_step}")
    target_steps = ceil(target_epochs * num_batches)
    return target_steps * num_rounds_per_step
