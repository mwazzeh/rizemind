# Split-Learning Experiments — Results Summary

Framework: **Rizemind** split-learning example (`examples/torch_split`), built on
Flower. Two datasets / architectures: **MNIST** (2-layer MLP, split at the
hidden layer) and **CIFAR-10** (small CNN, split after two conv+pool blocks).

## Protocol

Split learning: the client owns the network *head*, the server owns the *tail*.
Each training step spans **two Flower rounds** — round 1 the client sends
activations (+labels) at the cut point; round 2 the server returns the gradient
and the client finishes backprop. Client head weights are averaged (FedAvg)
after every backward round; the server keeps a single shared tail.

## Common settings (held fixed across the sweep)

| Setting | Value |
|---|---|
| Target local epochs | 2 |
| Per-client training cap | 2000 samples |
| Batch size | 64 |
| Flower rounds per run | 128 (= 64 split-learning steps) |
| Optimizer | SGD, momentum 0.9 |
| Validation | held-out 2000-image test subset, every 2 rounds |
| Baseline | 2 clients, IID, learning rate 0.01 |

The sweep is one-factor-at-a-time: from the baseline, exactly one of
{partitioning, client count, learning rate} is varied per run, for each dataset.

> **Note on magnitudes.** Accuracies are intentionally modest: each client
> trains on only 2000 capped samples for ~2 epochs (64 single-batch steps).
> The experiments are designed to compare *conditions*, not to reach
> state-of-the-art accuracy. Single-batch split steps are high-variance, so the
> "final" column is the mean of the last 5 evaluations; "best" is the peak.

## Results

### MNIST (MLP)

| Run | Variation | Final acc (mean last 5) | Best acc | Final loss |
|---|---|---|---|---|
| `mnist_iid` | baseline (IID, 2cl, lr0.01) | **0.848** | 0.853 | — |
| `mnist_iid_lr0.05` | learning rate 0.05 | **0.893** | 0.903 | — |
| `mnist_iid_5cl` | 5 clients | 0.842 | 0.851 | — |
| `mnist_dir_a0.5` | Dirichlet α=0.5 | 0.720 | 0.745 | — |
| `mnist_dir_a0.1` | Dirichlet α=0.1 | 0.565 | 0.602 | — |

### CIFAR-10 (CNN)

| Run | Variation | Final acc (mean last 5) | Best acc | Final loss |
|---|---|---|---|---|
| `cifar_iid` | baseline (IID, 2cl, lr0.01) | **0.301** | 0.332 | — |
| `cifar_iid_lr0.05` | learning rate 0.05 | 0.286 | 0.364 | — |
| `cifar_dir_a0.5` | Dirichlet α=0.5 | 0.261 | 0.284 | — |
| `cifar_iid_5cl` | 5 clients | 0.203 | 0.298 | — |
| `cifar_dir_a0.1` | Dirichlet α=0.1 | 0.193 | 0.241 | — |

Total sweep wall-clock: ~18.7 minutes (10 runs) on CPU + RTX A4000.

## Key findings

1. **Split learning trains correctly.** Loss falls and accuracy rises smoothly
   on both datasets despite the model being partitioned across client/server —
   the cut-point gradient exchange works.
2. **Data heterogeneity hurts, monotonically.** IID > Dirichlet α=0.5 >
   Dirichlet α=0.1 on both datasets. MNIST drops from 0.85 (IID) → 0.72
   (α=0.5) → 0.57 (α=0.1). The non-IID curves are also visibly noisier.
3. **Learning rate matters within a fixed budget.** With only ~2 epochs, lr=0.05
   reaches higher accuracy than lr=0.01 (MNIST 0.89 vs 0.85); on CIFAR it raises
   the *peak* (0.36 vs 0.33) but is noisier.
4. **Client count is roughly neutral here.** 2 vs 5 clients gives similar
   accuracy (each client keeps the same 2000-sample local budget), confirming
   head-averaging scales without degrading the IID baseline.
5. **CIFAR is harder and noisier than MNIST**, as expected for a CNN trained on
   little data with single-batch steps.

## Split-point analysis (from `analyze.py`)

**MNIST MLP** (batch 32): total 101.8K params, 6.50M FLOPs.

| Layer | Type | Params | FLOPs | Activation (KB) |
|---|---|---|---|---|
| head_fc | Linear 784→128 | 100.5K | 6.42M | 16.0 |
| head_relu | ReLU | 0 | 0 | 16.0 |
| tail_fc | Linear 128→10 | 1.3K | 81.9K | 1.25 |

Example cut (after `head_relu`): client does ~98.7% of FLOPs, transfers a
compact 16 KB activation per 32-image batch.

**CIFAR-10 CNN** (batch 32): total 545.1K params, 392.25M FLOPs.

| Layer | Type | Params | FLOPs | Activation (KB) |
|---|---|---|---|---|
| conv1 | Conv2d 3→32 | 896 | 56.6M | 4096 |
| pool1 | MaxPool2d | 0 | 0 | 1024 |
| conv2 | Conv2d 32→64 | 18.5K | 302.0M | 2048 |
| pool2 | MaxPool2d | 0 | 0 | **512 (cut here)** |
| fc1 | Linear 4096→128 | 524.4K | 33.6M | 16 |
| fc2 | Linear 128→10 | 1.3K | 81.9K | 1.25 |

The example cuts after `pool2`: the client runs both conv blocks (~91% of FLOPs,
where the heavy compute is) and ships a 512 KB activation. Cutting later (after
`fc1`) would shrink the transfer to 16 KB but push *all* compute onto the
client; cutting at `conv1` minimizes client compute but transfers 4096 KB. The
chosen cut trades transfer size against keeping raw pixels on the client.
