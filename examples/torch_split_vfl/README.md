# torch_split_vfl — Vertical Split Federated Learning

A minimal, runnable reference for **vertical** split federated learning (VFL)
on top of [Flower](https://flower.ai) and
[Rizemind](https://github.com/T-RIZE-Group/rizemind)'s
`VerticalSplitLearningStrategy`.

Each client holds the *same samples* but a *different subset of features*; the
server holds the labels and a small classifier tail. This is the horizontal
`torch_split` example's mirror image — there, clients hold *different samples*
of the same shape.

Two tasks are included:

* **MNIST** (`dataset="mnist"`) — image columns split into vertical pixel strips
  (a clean, visual illustration of the mechanics).
* **UCI Adult** (`dataset="adult"`) — a realistic tabular case where parties own
  *semantically different columns* (demographics vs. work/financial), which is
  what VFL is actually for. See [Tabular VFL: UCI Adult](#tabular-vfl-uci-adult).

> **Scope.** This is a v1 toy example: a faithful, end-to-end demonstration of
> the VFL *mechanics* (sample-aligned activations across K clients, per-client
> gradient slicing, no weight averaging). It is **not** a benchmark and **not**
> privacy-preserving — there is no PSI, secure aggregation, DP, or activation
> encryption. The server sees raw activations and cleartext labels. Treat
> outputs as a correctness check, not a private VFL protocol.

## Architecture

```
MNIST image (1, 28, 28) ── split vertically by column ──┐
        ▼                                               ▼
┌────────────────┐   ┌────────────────┐         ┌─────────────────┐
│  Client 0      │   │  Client 1      │   ...   │  Client K-1     │
│  cols [0, w)   │   │  cols [w, 2w)  │         │  cols [.., 28)  │
│  BottomMLP_0   │   │  BottomMLP_1   │         │  BottomMLP_{K-1}│
│  → act_0 (H,)  │   │  → act_1 (H,)  │         │  → act_{K-1}(H,)│
└──────┬─────────┘   └──────┬─────────┘         └────────┬────────┘
       └──────────── concat → (B, K*H) ──────────────────┘
                            ▼
                  ┌───────────────────┐
                  │   Server tail     │  (also holds the labels)
                  │  Linear→ReLU→Linear│
                  └─────────┬─────────┘
                            ▼  cross-entropy → backprop
                gradient sliced per client → returned to each client
```

## Vertical vs. horizontal

| | `torch_split` (horizontal) | `torch_split_vfl` (this example) |
|---|---|---|
| Each client holds | Same features, **different samples** | **Same samples**, different features |
| Bottom model | One shared head, FedAvg'd between rounds | One per client, **never averaged** |
| Sample alignment | Not required | **Required** — same ids per step on every party |
| Server input | One activation per sampled client | Concatenation of all K activations |
| Labels | At each client | At the server |

**Why no FedAvg.** Each bottom model encodes a *different* strip of pixels into
the joint representation, so the models are not interchangeable — even when
strip widths make their shapes match, averaging them would mix encoders of
disjoint feature regions. Each `BottomMLP_k` is trained only by the gradient
backprop produces for *its* slice of the concatenated activation. Accordingly,
`VerticalSplitLearningStrategy` does not wrap a `FedAvg` base strategy, and its
`evaluate()` loads each client's cached weights individually.

## How it works

**SL step ↔ Flower rounds.** One SL step = one forward + one backward round, so
`num-server-rounds = 20` runs 10 training steps (10 mini-batches per client). A
full epoch over `M` mini-batches needs `2*M` rounds.

**Sample alignment.** VFL only works if every party reads the *same sample ids*
in the same order each step. Instead of a per-step coordination protocol, all
parties derive the order from a shared constant (`VFL_SHUFFLE_SEED = 42` in
`task.py`). The server sends `sl_step` to every client each forward round, and
each party computes the same batch:

```python
epoch, batch_idx = divmod(sl_step, num_batches)
ids = torch.randperm(N, seed=VFL_SHUFFLE_SEED + epoch)[batch_idx*B : (batch_idx+1)*B]
```

Identical seed and `N` ⇒ identical ids on every party.

**Server holds the labels** because it runs the top model — the simplest VFL
layout, keeping the wire format to activations only. A production deployment
would instead route labels to an "active party" leader; that is out of scope
here.

**One step, round by round:**

1. **Forward (configure):** strategy ships `sl_step` to all K clients.
2. **Forward (client):** each client pulls its batch, runs `BottomMLP_k`,
   returns `activation_k` tagged with its `partition_id`. The
   `split_learning_mod` middleware marks the payload as an activation.
3. **Forward (aggregate):** strategy checks exactly K activations arrived with
   unique ids, sorts by `partition_id`, and calls `on_train_step`.
4. **Server:** concatenate the K activations → `(B, K*H)`, run the tail,
   compute cross-entropy against the aligned label batch, backprop to each
   `activation_k.grad`, return one gradient per client.
5. **Backward (client):** each client restores its saved input + weights,
   replays the forward pass, applies the received gradient, steps the
   optimizer, returns updated bottom weights.
6. **Backward (aggregate):** strategy caches each client's weights by
   `partition_id` and advances `sl_step`.
7. **Evaluate:** once all partitions have reported, the server rebuilds every
   bottom model from cache and runs the full pipeline on the held-out test set.

## Run

```bash
cd examples/torch_split_vfl

uv run -- flwr run .                                            # 2 clients (left/right halves)
uv run -- flwr run . local-4 --run-config 'min-available-clients=4'  # 4 clients (4 columns)
uv run -- flwr run . --run-config 'demo=true num-server-rounds=6'    # step-by-step logs
uv run -- flwr run . --run-config 'results-path="results.jsonl"'     # per-round JSONL metrics
```

When using `local-4`, also set `min-available-clients=4` — that is currently
the only way to tell the strategy how many vertical partitions (K) to expect,
and it must equal the federation's supernode count.

## Configuration

| Key | Default | Notes |
|-----|---------|-------|
| `dataset` | `"mnist"` | `"mnist"` (image strips, K configurable) or `"adult"` (tabular columns, K=2). |
| `num-server-rounds` | `20` | 2 rounds per training step. |
| `min-available-clients` | `2` | K — vertical participants. **Must equal `num-supernodes`.** |
| `batch-size` | `64` | Mini-batch shared by all parties. |
| `learning-rate` | `0.05` | Shared by every bottom + tail SGD. |
| `hidden-dim` | `64` | Per-client activation width at the cut. |
| `max-train-samples` | `2000` | Per-process cap for a fast smoke test (`0` = full split). |
| `eval-every` | `2` | Centralized eval every N rounds. |
| `eval-max-samples` | `1000` | Cap test images per eval. |
| `results-path` | `""` | When set, append per-round JSONL metrics. |
| `demo` | `false` | Per-step `[VSL-DEMO]` trace (useful for teaching, noisy on long runs). |

## What to expect

The default smoke run (≈4 SL steps, 512 samples, K=2) is too small to converge:
`val_acc` stays near random (~0.10) while `val_loss` drifts down from ~2.30.
The point is to confirm the round machinery runs end to end.

For real learning, run longer, e.g.:

```bash
uv run -- flwr run . --run-config 'num-server-rounds=40 max-train-samples=4000 learning-rate=0.1'
```

In reference runs this reaches `val_acc ≈ 0.75` / `val_loss ≈ 0.76` by step 20.
These are sanity checks, not benchmarks — the bottom MLPs and sample cap are
deliberately minimal.

## Tabular VFL: UCI Adult

`dataset="adult"` swaps MNIST's artificial pixel strips for a realistic tabular
split where each party owns genuinely different columns of the
[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) census data (predict
whether income > \$50K):

* **Party 0 — demographic:** age, sex, race, marital status, relationship,
  native country.
* **Party 1 — work / financial:** workclass, education, education-num,
  occupation, hours-per-week, capital gain/loss, fnlwgt.
* **Server:** holds the `income` label and the classifier tail.

This is a **K=2** task (the split is defined by the two column groups), so run
it on the default 2-supernode federation:

```bash
# Quick smoke run (~8 s): confirms the tabular pipeline trains end to end.
uv run -- flwr run . --run-config 'dataset="adult" num-server-rounds=20 max-train-samples=3000 eval-max-samples=1500'

# Longer run — learns past the majority-class baseline (~0.76) to val_acc ≈ 0.85.
uv run -- flwr run . --run-config 'dataset="adult" num-server-rounds=300 max-train-samples=8000 learning-rate=0.1 eval-every=10'
```

Preprocessing (in `task.py`) is deterministic and leakage-free: a fixed-seed
train/test split; numerical columns standardized with **train** mean/std;
categorical columns one-hot encoded against the **train** vocabulary (unseen
test categories → all-zeros); `?` kept as its own category. Each party's tensor
is built only from its own columns, so the label never reaches a client. Row
order is preserved, so the same `VFL_SHUFFLE_SEED` batching keeps all parties
aligned. The bottom MLP is reused as-is — only the per-party input width differs
(party 0 ≈ 62 features, party 1 ≈ 45 after encoding).

## Experiment sweep

`experiments/run_sweep.py` runs a one-factor-at-a-time sweep around a baseline
(K=2, lr=0.05, hidden-dim=64), varying the factors that matter for VFL — party
count (K ∈ {2, 4, 7}), learning rate, and cut width (hidden-dim) — and records
per-round metrics to `experiments/results/<id>.jsonl` plus a combined
`manifest.json`. `experiments/plot_results.py` turns those into comparison
figures.

Party count is swept **two ways**, because K is entangled with total cut
capacity (the server tail input is `K * hidden_dim`): one sweep keeps per-party
`hidden_dim` fixed (so total width grows with K — capacity-confounded), and one
keeps the total cut width ~constant by shrinking `hidden_dim` as K grows (which
isolates the effect of party count). The two are plotted separately; see
[`RESULTS_SUMMARY.md`](experiments/RESULTS_SUMMARY.md).

```bash
uv run python experiments/run_sweep.py            # full sweep (~8 min)
uv run python experiments/run_sweep.py --dry-run  # print the commands only
uv run python experiments/run_sweep.py --only mnist_k2
uv run python experiments/plot_results.py         # write plots to experiments/plots/
```

Every run uses the same round budget and sample cap so only the swept factor
changes. There is no IID/non-IID axis here — in VFL all parties share the same
samples — so heterogeneity is replaced by the K / learning-rate / cut-width
factors above.

## Tests

```bash
uv run pytest tests   # 27 fast unit tests
```

They cover strip layout, sample alignment across parties, model wiring, client
forward/backward state handling, and the Adult tabular preprocessing
(`test_adult.py`: feature-partition shapes, sample alignment, label safety, and
a short end-to-end training step — all offline on a synthetic table).

## Roadmap (design notes — not yet implemented)

* **CIFAR-10 quadrant split** — 4 parties, one 16×16 quadrant each, with small
  CNN bottoms (Conv→ReLU→Pool→flatten→Linear).
* **Per-party feature counts in the sweep** — extend the experiment harness with
  an Adult sweep (e.g. moving columns between parties, varying hidden-dim).

Both are additive at the `task.py` / `experiments/` level and leave the strategy
contract unchanged.
