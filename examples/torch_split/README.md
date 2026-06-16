# torch_split — Federated Split Learning on MNIST and CIFAR-10

Horizontal split learning over `SplitLearningStrategy`. The network is
partitioned at a *cut layer*: each client owns the first layers (the head) and
the server owns the rest (the tail). Clients never share raw weights for the
server's layers, and the server never sees the raw input — only the activations
at the cut point cross the wire.

The example trains across simulated clients on either dataset and exercises the
full forward/backward protocol, optional step-by-step logging, a companion tool
for choosing where to split a model, and a reproducible experiment sweep.

| `dataset` | Data | Architecture |
|---|---|---|
| `"mnist"` (default) | 28×28 grayscale, flattened to 784 | 2-layer MLP, split at the hidden layer |
| `"cifar10"` | 3×32×32 color | small CNN, split after two conv+pool blocks |

---

## How split learning works here

One **training step** spans **two Flower rounds**:

```
Step k
├─ Round 2k-1  (forward)
│    Client  ── runs head on a local mini-batch
│            ── sends activations + labels ───────────────▶  Server
│    Server  ── runs tail, computes loss, derives the
│               gradient at the cut point
│
└─ Round 2k    (backward)
     Server  ── sends each client its per-client gradient ─▶  Client
     Client  ── redoes the forward pass, applies the
                gradient, updates and persists head weights
```

The server keeps a single tail model that all clients share; client head
weights are averaged (FedAvg) after every backward round to form the global
head. Because a Flower client instance is recreated each round, the client
persists its head weights, current batch, and training cursor in
`context.state` so the backward round can reconstruct an identical forward
pass.

### The models

**MNIST** — a 2-layer MLP on flattened input (28×28 = 784):

```
Input (784,) ─▶ ClientHead: Linear(784→128) → ReLU ─▶ activations (128,)
                                  ╎ cut point ╎
activations  ─▶ ServerTail: Linear(128→10)         ─▶ logits ─▶ loss
```

**CIFAR-10** — a small CNN; the cut is after two conv+pool blocks:

```
Input (3,32,32) ─▶ ConvClientHead: [Conv→ReLU→Pool] ×2 ─▶ activations (64,8,8)
                                       ╎ cut point ╎
activations     ─▶ ConvServerTail: Flatten→Linear(4096→128)→ReLU→Linear(128→10)
```

Each client receives a partition of the training set (IID by default,
Dirichlet non-IID optional) with a local train/validation split. The
transformed data is materialized to tensors once per process, so fetching the
next mini-batch across rounds is an O(1) index lookup.

The split-learning wire contract is identical for both: one activation tensor
(plus labels) per forward round, one gradient tensor per backward round. Only
the tensor shapes differ.

---

## Quickstart

From this directory (`uv` resolves dependencies automatically — no manual
install step needed):

```bash
cd examples/torch_split
uv run -- flwr run .                              # MNIST (default)
uv run -- flwr run . --run-config 'dataset="cifar10"'   # CIFAR-10 CNN
```

You'll see centralized evaluation metrics (`val_loss`, `val_accuracy`,
`train_loss`) reported each round.

> **Note on accuracy.** The defaults (`num-server-rounds = 10`) run only
> **5 training steps**, i.e. 5 mini-batches per client — enough to confirm the
> pipeline learns (loss falls, accuracy rises), not to reach good accuracy.
> See [Training to convergence](#training-to-convergence) below.

### Demo mode

Turn on step-by-step logging to watch each forward and backward round:

```bash
uv run -- flwr run . --run-config 'demo=true'
```

Every demo line is prefixed with `[SL-DEMO]`, so you can isolate them:

```bash
uv run -- flwr run . --run-config 'demo=true' 2>&1 | grep '\[SL-DEMO\]'
```

Sample output:

```
[SL-DEMO] CLIENT node-6346  FORWARD  | epoch=0 batch=0  (32, 784) -> activation (32, 128), labels (32,)
[SL-DEMO] SERVER         BACKWARD | client ...6346  loss=2.3254  gradient (32, 128) ready
[SL-DEMO] CLIENT node-6346  BACKWARD | gradient (32, 128) applied, head updated + saved
```

---

## Configuration

All settings live under `[tool.flwr.app.config]` in `pyproject.toml` and can be
overridden per run with `--run-config 'key=value key2=value2'`.

| Key | Default | Description |
|---|---|---|
| `dataset` | `"mnist"` | `"mnist"` (MLP) or `"cifar10"` (CNN). Selects data *and* architecture. |
| `num-server-rounds` | `10` | Total Flower rounds. Two rounds = one training step, so this must be even. |
| `seed` | `42` | Seeds model init (tail + each head by `seed+pid`) for reproducible runs. Data partitioning uses `partition-seed`. Same seed+config ⇒ reproducible; different seed ⇒ independent trial. |
| `target-epochs` | `0.0` | When `> 0`, overrides `num-server-rounds` to cover roughly this many local epochs (see below). |
| `min-available-clients` | `2` | Clients required before a round starts. Set equal to the federation's supernode count. |
| `batch-size` | `32` | Mini-batch size for training and evaluation. |
| `learning-rate` | `0.01` | SGD learning rate (head and tail). |
| `val-ratio` | `0.10` | Fraction of each client partition held out for validation. |
| `max-train-samples` | `0` | Per-client training cap (`0` = full partition). Keeps full-epoch sweeps cheap. |
| `partitioner` | `"iid"` | `"iid"` or `"dirichlet"` (non-IID). |
| `partition-seed` | `42` | Seed for partitioning and shuffling. |
| `dirichlet-alpha` | `0.5` | Dirichlet concentration (lower = more skewed). Used when `partitioner="dirichlet"`. |
| `dirichlet-min-partition-size` | `10` | Minimum samples per partition for Dirichlet. |
| `dirichlet-self-balancing` | `false` | Rebalance Dirichlet partition sizes. |
| `hidden-dim` | `128` | Cut-point activation width (MLP) / tail hidden width (CNN). |
| `cut-layer` | `0` | **Informational** — the split is fixed by the head/tail classes in `task.py`. Use [`analyze.py`](#choosing-a-split-point) to decide where to cut, then edit those classes. |
| `eval-every` | `1` | Evaluate every N rounds (throttles centralized + distributed eval). |
| `eval-max-samples` | `0` | Cap on server-side eval images (`0` = full test split). A held-out subset speeds frequent evaluation. |
| `results-path` | `""` | When set, the server appends per-round metrics (`round`, `step`, `val_loss`, `val_accuracy`, `train_loss`) as JSON lines. |
| `demo` | `false` | Enable `[SL-DEMO]` step-by-step logging. |

The class/channel counts are derived from `dataset`, so they are no longer
separate config keys.

The number of simulated clients is set by the chosen *federation*. Two are
defined in `pyproject.toml`: `local-simulation` (2 supernodes, the default) and
`local-5` (5 supernodes). Select one with `flwr run . <federation>` and set
`min-available-clients` to match.

### Training to convergence

Two ways to train on more data:

```bash
# Explicit round count — 200 rounds = 100 steps per client
uv run -- flwr run . --run-config 'num-server-rounds=200'

# Or target a number of local epochs; rounds are derived from the
# largest client partition (each step = one mini-batch).
uv run -- flwr run . --run-config 'target-epochs=1.0'
```

When `target-epochs > 0`, the server computes the rounds needed for the largest
client partition to complete that many local epochs; smaller partitions wrap
and begin a new epoch slightly earlier.

### Non-IID data

```bash
uv run -- flwr run . --run-config 'partitioner="dirichlet" dirichlet-alpha=0.3'
```

---

## Choosing a split point

`analyze.py` profiles any PyTorch model layer by layer to help decide where to
cut. It reports per-layer forward time, activation (wire) size, FLOPs, and —
with `--training` — backward time and an estimate of client-side training
memory.

```bash
# Default: a 5-layer demo MLP, float32, batch=32, cpu
uv run python analyze.py

# Profile this example's MNIST model (784 → 128 → 10)
uv run python analyze.py --model torch_split

# Profile this example's CIFAR-10 CNN, with backward-pass + memory profiling
uv run python analyze.py --model torch_split_cnn --training

# Tables only, skip the PNG
uv run python analyze.py --no-plot
```

GPU timing is collected automatically when CUDA is available (disable with
`--no-gpu-timing`). The output includes a per-layer table, a per-cut-point
table, and recommended cuts for four objectives: minimum client compute,
minimum activation transfer, best compute balance, and minimum client memory.
A multi-panel PNG is written to `split_analysis.png` unless `--no-plot` is set.

To profile your own model from Python:

```python
import torch, torch.nn as nn
from torch_split.analysis import LayerProfiler

model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
sample = torch.randn(32, 64)

profiler = LayerProfiler(model, sample, device="cpu")
stats = profiler.profile()
recs = profiler.recommendations(stats)
```

---

## How it maps to the Rizemind library

This example is a thin application layer over
`rizemind.split_learning`:

| Library piece | Role |
|---|---|
| `SplitLearningStrategy` | Wraps a base Flower strategy (here `FedAvg`); alternates forward/backward phases and routes per-client gradients. |
| `split_learning_mod` | Client middleware that tags each round as forward or backward and marks activation payloads. |
| `serialization` | Converts tensors ↔ Flower `Parameters` for activations and gradients. |
| `SplitLearningConfig` | Declares the cut layer and rounds-per-step. |

The example supplies the model, data loading, and the `server_backward_fn`
(loss + gradient at the cut point); the strategy and middleware handle the
two-round choreography.

---

## Experiment sweep

`experiments/run_sweep.py` runs a one-factor-at-a-time sweep across both
datasets — varying data heterogeneity (IID / Dirichlet), client count, and
learning rate around a baseline — and records per-round metrics to
`experiments/results/<run_id>.jsonl` plus a combined `manifest.json`.
`experiments/plot_results.py` turns those into comparison figures.

```bash
uv run python experiments/run_sweep.py            # full sweep
uv run python experiments/run_sweep.py --dry-run  # print the commands only
uv run python experiments/run_sweep.py --only mnist_iid cifar_iid
uv run python experiments/plot_results.py         # write plots to experiments/plots/
```

Each run uses `target-epochs` with a per-client `max-train-samples` cap and a
throttled, subsampled evaluation (`eval-every`, `eval-max-samples`) so the
whole sweep finishes in minutes. Single-batch split steps are inherently
high-variance, so the summary chart reports the mean of the last few
evaluations rather than the single final point.

---

## Project layout

| File | Responsibility |
|---|---|
| `torch_split/task.py` | Dataset registry, models (MLP + CNN), data loading, partition caching, wire-format helpers |
| `torch_split/client.py` | Forward/backward client logic and `context.state` persistence |
| `torch_split/server.py` | Server backward function, centralized + distributed evaluation, metrics logging, round config |
| `torch_split/analysis.py` | Layer-profiler engine (`LayerProfiler`, `TrainingProfiler`) |
| `analyze.py` | CLI wrapper for the profiler |
| `experiments/` | Sweep runner, plotting script, and generated results/plots |
| `tests/` | Unit tests for the task, client, and analyzer |
| `pyproject.toml` | Flower app config, federations, and dependencies |

---

## Testing

```bash
uv run pytest          # all example tests
uv run pytest tests/test_analysis.py
```

The split-learning library itself is tested separately from the repo root:

```bash
uv run pytest tests/unit/py/rizemind/split_learning
```
