# torch_split_vfl — Vertical Split Learning

Vertical split federated learning (VFL): every client holds the *same samples*
but a *different subset of the features*, while the server holds the labels and a
small classifier on top. It is the mirror image of the horizontal `torch_split`
example, where clients instead hold different samples of the same shape.

Two tasks ship with the example:

* **MNIST** (`dataset="mnist"`) — images sliced into vertical pixel strips, one
  per client. Artificial, but it makes the mechanics easy to see.
* **UCI Adult** (`dataset="adult"`) — a tabular split where each party owns
  genuinely different columns (demographics vs. work/financial), which is the
  case VFL is actually meant for. See [Tabular VFL: UCI Adult](#tabular-vfl-uci-adult).

This is a v1 example, written to show the VFL mechanics end to end —
sample-aligned activations across K clients, per-client gradient slicing, and no
weight averaging. It is not a benchmark and not privacy-preserving: there is no
PSI, secure aggregation, DP, or activation encryption, and the server sees raw
activations and cleartext labels. Treat the outputs as a correctness check, not
a private protocol.

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
| Each client holds | Same features, different samples | Same samples, different features |
| Bottom model | One shared head, FedAvg'd between rounds | One per client, never averaged |
| Sample alignment | Not required | Required — same ids per step on every party |
| Server input | One activation per sampled client | Concatenation of all K activations |
| Labels | At each client | At the server |

Each bottom model encodes a different strip of pixels into the joint
representation, so the models are not interchangeable — even when matching strip
widths make their shapes line up, averaging them would mix encoders of disjoint
feature regions. Every `BottomMLP_k` is trained only by the gradient backprop
produces for its slice of the concatenated activation. That's why
`VerticalSplitLearningStrategy` does not wrap a `FedAvg` base strategy, and why
its `evaluate()` loads each client's cached weights individually.

## How it works

One SL step is one forward round plus one backward round, so
`num-server-rounds=20` runs 10 training steps (10 mini-batches per client). A
full epoch over `M` mini-batches needs `2*M` rounds.

VFL only works if every party reads the same sample ids in the same order each
step. Rather than coordinate per step, all parties derive the order from a shared
constant (`VFL_SHUFFLE_SEED = 42` in `task.py`). The server sends `sl_step` to
every client each forward round, and each party computes the same batch:

```python
epoch, batch_idx = divmod(sl_step, num_batches)
ids = torch.randperm(N, seed=VFL_SHUFFLE_SEED + epoch)[batch_idx*B : (batch_idx+1)*B]
```

Identical seed and `N` give identical ids on every party.

The server holds the labels because it runs the top model — the simplest layout,
and it keeps the wire format to activations only. A production deployment would
route labels to an "active party" leader instead; that's out of scope here.

A single step, round by round:

1. **Forward (configure):** the strategy ships `sl_step` to all K clients.
2. **Forward (client):** each client pulls its batch, runs `BottomMLP_k`, and
   returns `activation_k` tagged with its `partition_id`. The
   `split_learning_mod` middleware marks the payload as an activation.
3. **Forward (aggregate):** the strategy checks that exactly K activations
   arrived with unique ids, sorts by `partition_id`, and calls `on_train_step`.
4. **Server:** concatenate the K activations into `(B, K*H)`, run the tail,
   compute cross-entropy against the aligned label batch, backprop to each
   `activation_k.grad`, and return one gradient per client.
5. **Backward (client):** each client restores its saved input and weights,
   replays the forward pass, applies the received gradient, steps the optimizer,
   and returns updated bottom weights.
6. **Backward (aggregate):** the strategy caches each client's weights by
   `partition_id` and advances `sl_step`.
7. **Evaluate:** once every partition has reported, the server rebuilds all
   bottom models from cache and runs the full pipeline on the held-out test set.

## Run

```bash
cd examples/torch_split_vfl

uv run -- flwr run .                                            # 2 clients (left/right halves)
uv run -- flwr run . local-4 --run-config 'min-available-clients=4'  # 4 clients (4 columns)
uv run -- flwr run . --run-config 'demo=true num-server-rounds=6'    # step-by-step logs
uv run -- flwr run . --run-config 'results-path="results.jsonl"'     # per-round JSONL metrics
uv run -- flwr run . --run-config 'seed=123'                         # reproducible independent trial
```

With `local-4`, also set `min-available-clients=4`. That is currently the only
way to tell the strategy how many vertical partitions (K) to expect, and it must
equal the federation's supernode count.

### Genuine federated feature ablations (Adult)

The `active-parties` config selects which tabular feature groups are active, so
each ablation runs through the real VFL training path rather than a centralized
proxy. The number of active groups must equal `min-available-clients`, so a
single-party ablation uses the `local-1` federation:

```bash
# Both feature groups (the default K=2 semantic split)
uv run -- flwr run . --run-config 'dataset="adult" min-available-clients=2 active-parties="0,1"'

# Party 0 only (demographic features) — genuine 1-party VFL
uv run -- flwr run . local-1 --run-config 'dataset="adult" min-available-clients=1 active-parties="0"'

# Party 1 only (work/financial features)
uv run -- flwr run . local-1 --run-config 'dataset="adult" min-available-clients=1 active-parties="1"'
```

Excluded groups contribute no information — their party is simply not
instantiated — and the label stays at the server in every case. Invalid
selections (out-of-range id, duplicate, wrong count) fail with a clear error.

## Configuration

| Key | Default | Notes |
|-----|---------|-------|
| `dataset` | `"mnist"` | `"mnist"` (image strips, K configurable) or `"adult"` (tabular columns, K=2). |
| `seed` | `42` | Seeds model init (tail + each bottom by `seed+pid`) and the batch shuffle. Same seed+config gives a reproducible run; a different seed gives an independent trial. Replaces the old external shim. |
| `active-parties` | `""` | Tabular feature-group selection, e.g. `"0"`, `"1"`, `"0,1"`. Empty/`"all"` = every group. Count must equal `min-available-clients`. |
| `label-private` | `false` | When `true`, labels and the top model stay at `label-holder-party`; the coordinator never sees labels. Uses 3 rounds/step (set `num-server-rounds` to a multiple of 3). |
| `label-holder-party` | `0` | Partition id (in `[0, min-available-clients)`) that owns the labels and top model in label-private mode. |
| `num-server-rounds` | `20` | 2 rounds per training step (3 in label-private mode). |
| `min-available-clients` | `2` | K — vertical participants. Must equal `num-supernodes`. |
| `batch-size` | `64` | Mini-batch shared by all parties. |
| `learning-rate` | `0.05` | Shared by every bottom + tail SGD. |
| `hidden-dim` | `64` | Per-client activation width at the cut. |
| `max-train-samples` | `2000` | Per-process cap for a fast smoke test (`0` = full split). |
| `eval-every` | `2` | Centralized eval every N rounds. |
| `eval-max-samples` | `1000` | Cap test images per eval. |
| `results-path` | `""` | When set, append per-round JSONL metrics and write a `*.summary.json` (schema + final metrics + telemetry) at the end. |
| `demo` | `false` | Per-step `[VSL-DEMO]` trace — handy for teaching, noisy on long runs. |

## Reproducibility, metrics & telemetry

Set `seed` to make a run fully reproducible: it seeds Python/NumPy/PyTorch
(CPU+CUDA) for model initialisation and the per-epoch batch shuffle. The server
tail uses `seed`; each bottom uses `seed + partition_id`, so the inits are
distinct but reproducible. Every party shares `seed` for the shuffle, which keeps
sample alignment intact. No external `PYTHONPATH`/`sitecustomize` shim is needed,
and the effective seed is recorded in every result file. (The Adult train/test
split keeps its own fixed seed so ablations and references stay comparable.)

When `results-path` is set, each per-round JSONL record carries `val_accuracy`,
`val_loss`, and `train_loss` (unchanged for compatibility) plus `precision_macro`,
`recall_macro`, `f1_macro`, `balanced_accuracy`, and, for binary/Adult,
`precision`, `recall`, `f1`, and `roc_auc`. All come from the actual VFL model's
predictions on the held-out test set, never a centralized proxy, and edge cases
(zero-division, single class) are handled safely.

The final `*.summary.json` also includes a `communication` block with per-party
and aggregate activation/gradient payload-byte estimates
(`element_count × element_size`, not transport bytes), cumulative payload,
per-step means, and server forward/backward/eval timings plus total
`wall_clock_s`. The schema additionally records `seed`, `configured_parties`,
`active_party_count`, `active_parties`, `party_feature_dims`, `total_cut_width`,
`num_steps`, optimizer, and so on.

Telemetry records only sizes and timings — never activation/gradient contents,
labels, or feature values. In the default (non-label-private) mode the server
still sees raw activations and cleartext labels.

## Label-private mode (label confidentiality from the coordinator)

By default the coordinator holds the labels and the top model. Label-private mode
(`label-private=true`) moves both to a single configured `label-holder-party`, so
labels never reach, are stored by, or are logged at the Flower coordinator.

Each step takes three rounds instead of two:

1. **collect** — every party runs its bottom and returns an activation.
2. **compute** — the coordinator routes the joint activation (no labels) to the
   label holder only, which runs the top model against its local labels,
   backpropagates, and returns one gradient per party plus aggregate scalar
   metrics (loss, accuracy, F1, …). No labels or predictions come back.
3. **distribute** — each party receives its gradient and updates its bottom.

Evaluation stays label-free at the coordinator: it builds the joint test
activation from cached bottom weights and test features (never labels), ships it
to the holder, and the holder computes all metrics against its local test labels
and returns aggregate scalars only.

```bash
# Adult, both parties, party 0 is the label holder (300 rounds = 100 steps)
uv run -- flwr run . --run-config 'dataset="adult" min-available-clients=2 label-private=true label-holder-party=0 num-server-rounds=300 learning-rate=0.1'

# Adult party-1-only ablation, label-private (K=1; the party is its own holder)
uv run -- flwr run . local-1 --run-config 'dataset="adult" min-available-clients=1 active-parties="1" label-private=true label-holder-party=0 num-server-rounds=300 learning-rate=0.1'

# MNIST K=4, label holder = party 2
uv run -- flwr run . local-4 --run-config 'min-available-clients=4 hidden-dim=32 label-private=true label-holder-party=2 num-server-rounds=240'
```

The `*.summary.json` records `mode="vfl-label-private"`, `labels_at_server=false`,
`label_holder_party`, and a `communication.routing_bytes` breakdown of the extra
coordinator↔holder traffic (`repr_downlink`, `joint_grad_uplink`).

What this protects, and what it doesn't: it keeps the label tensor off the
coordinator and the passive parties. It does not make the system
privacy-preserving — activations and gradients are still exchanged in the clear,
and labels can often be inferred from gradients. It offers no protection against
activation/model inversion, membership inference, malicious parties, collusion,
traffic analysis, or transport interception, and applies no DP, encryption, PSI,
or secure aggregation.

## Cut-gradient privacy (clipped Gaussian perturbation)

Hiding the label tensor is not the same as hiding label information: for binary
cross-entropy the sign/direction of the cut gradient encodes the label, so an
honest-but-curious coordinator can reconstruct labels from the gradients it
routes. On our Adult benchmark a trivial attacker (the fraction of positive
gradient coordinates) and a logistic-regression shadow attacker both reach
ROC-AUC ≈ 1.0 on unprotected gradients.

Optional cut-gradient privacy clips and masks the gradient at the label holder,
before any coordinator-bound serialization. For the joint per-sample gradient
matrix `G` (shape `B × K·H`):

1. compute each sample row's L2 norm `‖G_i‖`;
2. clip each row to the bound `C`: `G_i ← G_i · min(1, C/‖G_i‖)`;
3. add Gaussian noise to each released per-sample row independently:
   `G̃_i = clip(G_i) + N(0, σ² I)` with `σ = gradient-noise-multiplier · C`;
4. split the protected `G̃` into party slices and serialize.

No clean gradient copy is ever serialized, logged, or stored by the coordinator.
The noise is added independently to each released per-sample row — not to a sum
or an average — because the protocol needs a usable per-sample gradient for every
party.

Configuration keys:

| key | meaning |
|---|---|
| `gradient-privacy-mode` | `none` (default) · `clip` · `gaussian` |
| `gradient-clip-norm` | per-sample L2 bound `C` (`gaussian`/`clip`) |
| `gradient-noise-multiplier` | `z`; noise `σ = z · C` (`gaussian` only) |
| `privacy-delta` | δ recorded for future accounting (no ε computed) |
| `privacy-rng-mode` | `research-seeded` (reproducible) · `secure` (OS entropy, no reusable seed) |

```bash
# Adult, clip + Gaussian noise (z=1.0) on the cut gradients, party 0 holds labels
uv run -- flwr run . --run-config 'dataset="adult" min-available-clients=2 label-private=true label-holder-party=0 num-server-rounds=300 learning-rate=0.1 gradient-privacy-mode="gaussian" gradient-clip-norm=0.006 gradient-noise-multiplier=1.0'
```

On binary tasks the label holder also runs a label-inference attack benchmark on
the coordinator-visible gradients at eval rounds, reporting aggregate attack
metrics (Attack A heuristic + Attack B shadow, with majority/chance references)
into `*.summary.json` — never labels or raw gradients. Clip/noise diagnostics
(norm percentiles, clip fraction, σ, SNR) and the privacy config are recorded
under `communication.privacy_diagnostics_*` and the schema (v3).

This is clipped Gaussian gradient perturbation, not formal differential privacy.
No audited accountant backs an ε under the actual sampling and composition, so
the schema sets `formal_dp_claim=false` and `epsilon=null`; the parameters needed
for later accounting (clip bound, σ, δ, batch/sample rate, effective steps) are
recorded. Any reduction in a specific stated attack is reported as exactly that —
attack resistance against the honest-but-curious coordinator — and is not a
guarantee against activation inversion, membership inference, malicious/colluding
parties, repeated-query composition, or traffic analysis. Clipping alone does not
reduce this leakage (it bounds magnitude, not the label-correlated sign pattern);
only the Gaussian noise does. Deterministic `research-seeded` runs are for
reproducible experiments — use `secure` RNG for anything resembling a deployment.
See `THREAT_MODEL.md` in the Phase-4 artifacts for the full threat model.

## What to expect

The default smoke run (≈4 SL steps, 512 samples, K=2) is too small to converge:
`val_acc` stays near random (~0.10) while `val_loss` drifts down from ~2.30. It
exists to confirm the round machinery runs end to end.

For real learning, run longer:

```bash
uv run -- flwr run . --run-config 'num-server-rounds=40 max-train-samples=4000 learning-rate=0.1'
```

In reference runs this reaches `val_acc ≈ 0.75` / `val_loss ≈ 0.76` by step 20.
These are sanity checks, not benchmarks — the bottom MLPs and sample cap are
deliberately minimal.

## Tabular VFL: UCI Adult

`dataset="adult"` swaps MNIST's artificial pixel strips for a realistic tabular
split where each party owns genuinely different columns of the
[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) census data (predicting
whether income > \$50K):

* **Party 0 — demographic:** age, sex, race, marital status, relationship,
  native country.
* **Party 1 — work / financial:** workclass, education, education-num,
  occupation, hours-per-week, capital gain/loss, fnlwgt.
* **Server:** holds the `income` label and the classifier tail.

The two column groups define a K=2 task, so run it on the default 2-supernode
federation:

```bash
# Quick smoke run (~8 s): confirms the tabular pipeline trains end to end.
uv run -- flwr run . --run-config 'dataset="adult" num-server-rounds=20 max-train-samples=3000 eval-max-samples=1500'

# Longer run — learns past the majority-class baseline (~0.76) to val_acc ≈ 0.85.
uv run -- flwr run . --run-config 'dataset="adult" num-server-rounds=300 max-train-samples=8000 learning-rate=0.1 eval-every=10'
```

Preprocessing (in `task.py`) is deterministic and leakage-free: a fixed-seed
train/test split; numerical columns standardized with train mean/std;
categorical columns one-hot encoded against the train vocabulary (unseen test
categories become all-zeros); `?` kept as its own category. Each party's tensor
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

Party count is swept two ways, because K is entangled with total cut capacity
(the server tail input is `K * hidden_dim`): one sweep keeps per-party
`hidden_dim` fixed, so total width grows with K (capacity-confounded), and the
other keeps total cut width roughly constant by shrinking `hidden_dim` as K grows,
which isolates the effect of party count. The two are plotted separately; see
[`RESULTS_SUMMARY.md`](experiments/RESULTS_SUMMARY.md).

```bash
uv run python experiments/run_sweep.py            # full sweep (~8 min)
uv run python experiments/run_sweep.py --dry-run  # print the commands only
uv run python experiments/run_sweep.py --only mnist_k2
uv run python experiments/plot_results.py         # write plots to experiments/plots/
```

Every run uses the same round budget and sample cap, so only the swept factor
changes. There is no IID/non-IID axis here — in VFL all parties share the same
samples — so heterogeneity is replaced by the K / learning-rate / cut-width
factors above.

## Tests

```bash
uv run pytest tests   # 68 fast unit tests
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
