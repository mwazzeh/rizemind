# VFL sweep — results summary

One-factor-at-a-time sweep on MNIST around a baseline of K=2, lr=0.05,
hidden-dim=64, 60 Flower rounds (30 SL steps), 4000-sample cap, evaluated on a
2000-image held-out subset. Reproduce with:

```bash
uv run python experiments/run_sweep.py
uv run python experiments/plot_results.py
```

Accuracy below is the mean of the last 5 evaluations — single-batch split steps
are high-variance, so the last point alone is noisy. Each run took ~55 s. The
numbers are from one unseeded run and vary by ±0.02–0.03 between repeats.

| Run | K | lr | hidden | Total cut width | Final val_acc (smooth) | Best |
|-----|---|----|--------|-----------------|------------------------|------|
| `mnist_k2` (baseline) | 2 | 0.05 | 64 | 128 | 0.682 | 0.747 |
| **Party count — fixed per-party hidden (growing total width)** ||||||
| `mnist_k4` | 4 | 0.05 | 64 | 256 | 0.743 | 0.766 |
| `mnist_k7` | 7 | 0.05 | 64 | 448 | 0.762 | 0.785 |
| **Party count — fixed total cut width (~128)** ||||||
| `mnist_ftw_k4` | 4 | 0.05 | 32 | 128 | 0.616 | 0.651 |
| `mnist_ftw_k7` | 7 | 0.05 | 18 | 126 | 0.497 | 0.581 |
| **Learning rate** ||||||
| `mnist_lr0.02` | 2 | 0.02 | 64 | 128 | 0.454 | 0.467 |
| `mnist_lr0.1` | 2 | 0.10 | 64 | 128 | 0.673 | 0.797 |
| **Cut width (hidden-dim)** ||||||
| `mnist_h32` | 2 | 0.05 | 32 | 64 | 0.560 | 0.613 |
| `mnist_h128` | 2 | 0.05 | 128 | 256 | 0.783 | 0.819 |

## Takeaways

Cut width (hidden-dim) dominates accuracy. Widening the per-client activation
32 → 64 → 128 lifts accuracy 0.560 → 0.682 → 0.783, monotonically. The cut is the
representational bottleneck: each bottom MLP has to compress its strip into
`hidden_dim` features before the server sees anything, so this is the first knob
to turn — at the cost of more bytes on the wire per round.

Party count needs the two sweeps read separately, because they say different
things. The number of parties (K) is entangled with total cut capacity: the
server tail input is the concatenation of all K activations (`K * hidden_dim`).
So we ran two K-sweeps:

* Fixed per-party `hidden_dim=64` (`mnist_k4`, `mnist_k7`): total cut width grows
  with K (128 → 256 → 448) and accuracy rises (0.682 → 0.743 → 0.762). This is
  capacity-confounded — more parties also means a wider server input — so it does
  not show that more parties help.

* Fixed total cut width ~128 (`mnist_ftw_k4`, `mnist_ftw_k7`): `hidden_dim` is
  shrunk as K grows (64 → 32 → 18) to hold `K * hidden_dim` roughly constant.
  Here accuracy drops with K (0.682 → 0.616 → 0.497).

What this supports, and what it doesn't:

* Increasing K did not break the VFL pipeline: every party count trained
  end-to-end and learned.
* Under the default fixed per-party setting, performance held steady or improved
  as K grew.
* That does not prove more VFL parties improve accuracy. The gain in the first
  sweep is consistent with simply having a larger total cut representation, not
  with the party count itself.
* When total cut capacity is held fixed (second sweep), more parties is worse
  here — splitting the same ~128-dim budget across more, narrower strips (down to
  a tiny 18-dim bottleneck at K=7) hurts. MNIST columns are also highly
  correlated, so finer vertical splits add little independent signal.

Learning rate: 0.05 sits in the right range for this budget. lr=0.02 is too slow
to converge in 30 steps (0.454); lr=0.05 (0.682) and lr=0.1 (0.673, but noisier —
best 0.797) are comparable.

## Caveats

* Toy scale — 4000 training samples, 30 SL steps, single-Linear bottom MLPs.
  These are *relative* comparisons, not benchmark numbers.
* No IID/non-IID axis exists in VFL: all parties share the same samples by
  construction, so heterogeneity is replaced by the K / lr / cut-width factors.
* Single-batch steps are high-variance; treat differences under ~0.02–0.03 as
  noise, and the two party-count sweeps as the controlled comparison rather than
  the raw K=2/4/7 accuracy numbers.

## Tabular VFL — UCI Adult (reference, not swept)

The sweep above is MNIST only. The `dataset="adult"` task (K=2: demographic vs.
work/financial columns, label at the server) is not part of the sweep, but as a
reference point, a single run:

```bash
uv run -- flwr run . --run-config 'dataset="adult" num-server-rounds=300 \
    max-train-samples=8000 learning-rate=0.1 eval-every=10'
```

reaches **val_acc ≈ 0.85** (val_loss ≈ 0.32), clearly above the ~0.76
majority-class baseline (Adult is ~24% positive). A short smoke run (20 rounds,
3000 samples) trains end-to-end in ~8 s but stays at the majority baseline — too
few steps to start predicting the minority class. These are sanity checks, not
tuned benchmarks.
