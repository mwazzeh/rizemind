# torch_split

A federated split learning example on MNIST using Rizemind's `SplitLearningStrategy` with Flower.

---

## How it works

The model is split at a cut layer: the client runs the first part of the network on its local data and sends the intermediate activations to the server. The server finishes the forward pass, computes the loss, and sends the gradient back so the client can complete backpropagation. This takes two Flower rounds per training step.

```
Round 1 (forward)
  Client  →  runs head layers on local batch
          →  sends activations + labels to server
  Server  →  runs tail layers, computes loss and gradient

Round 2 (backward)
  Server  →  sends per-client gradient back
  Client  →  redoes forward pass, applies gradient, updates head weights
```

## Model

A simple MLP on flattened MNIST (28×28 = 784 inputs):

```
Input (784,)  →  ClientHead: Linear(784→128) + ReLU  →  activations (128,)
                                   ↑ cut point ↑
activations   →  ServerTail: Linear(128→10)           →  logits → loss
```

Each client gets a local partition of MNIST (train/val split, IID by default). The transformed data is materialized once per process so batch access stays fast across rounds.

---

## Running

```bash
cd examples/torch_split
uv run -- flwr run .
```

To see step-by-step logging of what each client and the server are doing during training:

```bash
uv run -- flwr run . --run-config 'demo=true'
```

Demo lines are prefixed with `[SL-DEMO]` so you can filter them easily:

```bash
uv run -- flwr run . --run-config 'demo=true' 2>&1 | grep '\[SL-DEMO\]\|ROUND\|SUMMARY'
```

---

## Configuration

All settings live under `[tool.flwr.app.config]` in `pyproject.toml`:

```toml
num-server-rounds = 10   # 10 rounds = 5 training steps (2 rounds each)
target-epochs     = 0.0  # set > 0 to auto-compute rounds from local epoch count
batch-size        = 32
learning-rate     = 0.01
partitioner       = "iid"   # or "dirichlet" for non-IID data
partition-seed    = 42
cut-layer         = 0    # index of the last client-side layer
hidden-dim        = 128  # size of the activation at the cut point
num-classes       = 10
val-ratio         = 0.10
demo              = false
```

When `target-epochs > 0`, the number of rounds is derived from the largest client partition so that every client completes roughly that many local epochs.

---

## Split-point analysis

`analyze.py` profiles any PyTorch model layer by layer to help decide where to put the cut. It measures forward time, activation size, FLOPs, and optionally backward time and memory.

```bash
# Default: DemoMLP, float32, batch=32, cpu
uv run python analyze.py

# Match the model used in this example
uv run python analyze.py --model torch_split

# CNN with float16 and backward profiling
uv run python analyze.py --model conv --dtype float16 --training

# Tables only, skip the PNG
uv run python analyze.py --no-plot
```

GPU timing is collected automatically when CUDA is available. The output includes a per-layer table, a cut-point summary, and three recommendations: minimize client compute, minimize activation transfer size, and balance load between client and server.

To profile your own model:

```python
from torch_split.analysis import LayerProfiler
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
sample = torch.randn(32, 64)

profiler = LayerProfiler(model, sample, device="cpu")
stats = profiler.profile()
recs  = profiler.recommendations(stats)
```

---

## Key files

| File | What it does |
|---|---|
| `torch_split/task.py` | Model definitions, data loading, partition caching |
| `torch_split/client.py` | Client forward + backward logic, context.state persistence |
| `torch_split/server.py` | Server backward function, evaluation, round configuration |
| `torch_split/analysis.py` | Layer profiler engine |
| `analyze.py` | CLI wrapper for the profiler |
| `../../src/py/rizemind/split_learning/` | Core library: strategy, middleware, serialization |
