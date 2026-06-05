"""Run the torch_split experiment sweep and collect per-round metrics.

This drives ``flwr run`` once per configuration, pointing the server's
``results-path`` at a per-run JSON-lines file. The matrix is a one-factor-at-
a-time (OFAT) sweep around a baseline (2 clients, lr=0.01, IID), covering both
datasets/architectures (MNIST MLP, CIFAR-10 CNN).

Usage::

    uv run python experiments/run_sweep.py            # full sweep
    uv run python experiments/run_sweep.py --only mnist_iid cifar_iid
    uv run python experiments/run_sweep.py --dry-run   # print commands only

Each run writes ``experiments/results/<id>.jsonl``. A combined
``experiments/results/manifest.json`` records every run's config, status,
wall-clock time, and final metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Per-client training cap so "2 local epochs" stays cheap and comparable
# across runs (each client sees the same amount of local data).
MAX_TRAIN_SAMPLES = 2000
TARGET_EPOCHS = 2.0
BATCH_SIZE = 64
PARTITION_SEED = 42
EVAL_EVERY = 2          # record metrics once per SL step (every 2 Flower rounds)
EVAL_MAX_SAMPLES = 2000  # held-out test subset for fast, frequent evaluation

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
RESULTS_DIR = HERE / "results"


@dataclass
class Run:
    """One sweep configuration."""

    id: str
    dataset: str            # "mnist" | "cifar10"
    label: str              # human-readable label for plots
    group: str              # comparison group this run belongs to
    partitioner: str = "iid"
    dirichlet_alpha: float = 0.5
    clients: int = 2
    learning_rate: float = 0.01
    # Filled after running
    status: str = "pending"
    wall_seconds: float = 0.0
    rounds: int = 0
    final_val_accuracy: float = float("nan")       # last recorded eval point
    final_val_loss: float = float("nan")
    final_val_accuracy_smooth: float = float("nan")  # mean of last 5 eval points
    best_val_accuracy: float = float("nan")          # max over all eval points
    extra: dict = field(default_factory=dict)

    @property
    def federation(self) -> str:
        return "local-5" if self.clients == 5 else "local-simulation"

    @property
    def results_file(self) -> Path:
        return RESULTS_DIR / f"{self.id}.jsonl"

    def run_config(self) -> str:
        parts = [
            f'dataset="{self.dataset}"',
            f'partitioner="{self.partitioner}"',
            f"dirichlet-alpha={self.dirichlet_alpha}",
            f"learning-rate={self.learning_rate}",
            f"min-available-clients={self.clients}",
            f"target-epochs={TARGET_EPOCHS}",
            f"max-train-samples={MAX_TRAIN_SAMPLES}",
            f"batch-size={BATCH_SIZE}",
            f"partition-seed={PARTITION_SEED}",
            f"eval-every={EVAL_EVERY}",
            f"eval-max-samples={EVAL_MAX_SAMPLES}",
            f'results-path="{self.results_file}"',
        ]
        return " ".join(parts)


def build_matrix() -> list[Run]:
    """Return the OFAT sweep: baseline + one varied factor per run, x2 datasets."""
    runs: list[Run] = []
    for ds, tag in (("mnist", "MNIST-MLP"), ("cifar10", "CIFAR10-CNN")):
        pfx = "mnist" if ds == "mnist" else "cifar"
        runs += [
            # Baseline + partitioning sweep
            Run(f"{pfx}_iid", ds, f"{tag} IID", "partitioning",
                partitioner="iid"),
            Run(f"{pfx}_dir_a0.5", ds, f"{tag} Dirichlet α=0.5", "partitioning",  # noqa: RUF001
                partitioner="dirichlet", dirichlet_alpha=0.5),
            Run(f"{pfx}_dir_a0.1", ds, f"{tag} Dirichlet α=0.1", "partitioning",  # noqa: RUF001
                partitioner="dirichlet", dirichlet_alpha=0.1),
            # Client-count sweep (baseline IID, 5 clients)
            Run(f"{pfx}_iid_5cl", ds, f"{tag} IID 5 clients", "clients",
                partitioner="iid", clients=5),
            # Learning-rate sweep (baseline IID, lr=0.05)
            Run(f"{pfx}_iid_lr0.05", ds, f"{tag} IID lr=0.05", "lr",
                partitioner="iid", learning_rate=0.05),
        ]
    return runs


def execute(run: Run, dry_run: bool) -> None:
    cmd = [
        "uv", "run", "flwr", "run", ".", run.federation,
        "--run-config", run.run_config(),
    ]
    print(f"\n=== {run.id} ({run.label}) ===")
    print("  " + " ".join(cmd[:6]) + f" --run-config '{run.run_config()}'")
    if dry_run:
        run.status = "dry-run"
        return

    run.results_file.unlink(missing_ok=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True)
    run.wall_seconds = round(time.perf_counter() - start, 1)

    if proc.returncode != 0:
        run.status = "failed"
        log_path = RESULTS_DIR / f"{run.id}.stderr.log"
        log_path.write_text(proc.stdout + "\n" + proc.stderr)
        print(f"  FAILED (exit {proc.returncode}) — see {log_path}")
        return

    records = _load_jsonl(run.results_file)
    if not records:
        run.status = "no-metrics"
        print("  WARNING: completed but produced no metric records")
        return

    accs = [r["val_accuracy"] for r in records]
    last = records[-1]
    tail = accs[-5:]
    run.rounds = last.get("round", 0)
    run.final_val_accuracy = last.get("val_accuracy", float("nan"))
    run.final_val_loss = last.get("val_loss", float("nan"))
    run.final_val_accuracy_smooth = round(sum(tail) / len(tail), 4)
    run.best_val_accuracy = round(max(accs), 4)
    run.status = "ok"
    print(
        f"  ok in {run.wall_seconds}s — {run.rounds} rounds, "
        f"final(smooth) val_acc={run.final_val_accuracy_smooth:.4f}  "
        f"best={run.best_val_accuracy:.4f}"
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Run only these run ids.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, do not run.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = build_matrix()
    if args.only:
        runs = [r for r in runs if r.id in set(args.only)]
        if not runs:
            raise SystemExit(f"no runs matched {args.only}")

    total_start = time.perf_counter()
    for run in runs:
        execute(run, args.dry_run)

    manifest = {
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "target_epochs": TARGET_EPOCHS,
        "batch_size": BATCH_SIZE,
        "partition_seed": PARTITION_SEED,
        "total_wall_seconds": round(time.perf_counter() - total_start, 1),
        "runs": [asdict(r) for r in runs],
    }
    manifest_path = RESULTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nManifest written to {manifest_path}")
    ok = sum(1 for r in runs if r.status == "ok")
    print(f"Completed {ok}/{len(runs)} runs in {manifest['total_wall_seconds']}s")


if __name__ == "__main__":
    main()
