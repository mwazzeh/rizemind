"""Run the torch_split_vfl experiment sweep and collect per-round metrics.

This drives ``flwr run`` once per configuration, pointing the server's
``results-path`` at a per-run JSON-lines file. The matrix is a one-factor-at-
a-time (OFAT) sweep around a baseline (K=2, lr=0.05, hidden-dim=64).

Vertical FL has no IID/non-IID axis — every party shares the same samples — so
the factors swept here are the ones that actually matter for VFL:

* **K (party count)** — how many vertical feature strips the image is split
  into. More parties means narrower strips and a wider concatenated activation.
* **learning rate** — shared by every bottom + tail SGD.
* **cut width (hidden-dim)** — the per-client activation size at the cut.

Usage::

    uv run python experiments/run_sweep.py            # full sweep
    uv run python experiments/run_sweep.py --only mnist_k2
    uv run python experiments/run_sweep.py --dry-run  # print commands only

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

# Shared budget so every run does the same number of SL steps and sees the same
# amount of data — only the swept factor changes between runs.
NUM_SERVER_ROUNDS = 60  # 30 SL steps (2 Flower rounds per step)
MAX_TRAIN_SAMPLES = 4000
BATCH_SIZE = 64
EVAL_EVERY = 2  # record metrics once per SL step (every 2 Flower rounds)
EVAL_MAX_SAMPLES = 2000  # held-out test subset for fast, frequent evaluation

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
RESULTS_DIR = HERE / "results"


@dataclass
class Run:
    """One sweep configuration."""

    id: str
    label: str  # human-readable label for plots
    group: str  # comparison group this run belongs to
    clients: int = 2  # K — number of vertical parties
    learning_rate: float = 0.05
    hidden_dim: int = 64
    dataset: str = "mnist"
    # Filled after running
    status: str = "pending"
    wall_seconds: float = 0.0
    rounds: int = 0
    final_val_accuracy: float = float("nan")  # last recorded eval point
    final_val_loss: float = float("nan")
    final_val_accuracy_smooth: float = float("nan")  # mean of last 5 eval points
    best_val_accuracy: float = float("nan")  # max over all eval points
    extra: dict = field(default_factory=dict)

    @property
    def federation(self) -> str:
        # K=2 is the default federation; K=4 / K=7 have named federations.
        return "local-simulation" if self.clients == 2 else f"local-{self.clients}"

    @property
    def results_file(self) -> Path:
        return RESULTS_DIR / f"{self.id}.jsonl"

    def run_config(self) -> str:
        parts = [
            f'dataset="{self.dataset}"',
            f"learning-rate={self.learning_rate}",
            f"min-available-clients={self.clients}",
            f"hidden-dim={self.hidden_dim}",
            f"num-server-rounds={NUM_SERVER_ROUNDS}",
            f"max-train-samples={MAX_TRAIN_SAMPLES}",
            f"batch-size={BATCH_SIZE}",
            f"eval-every={EVAL_EVERY}",
            f"eval-max-samples={EVAL_MAX_SAMPLES}",
            f'results-path="{self.results_file}"',
        ]
        return " ".join(parts)


def build_matrix() -> list[Run]:
    """Return the OFAT sweep: baseline (K=2) + one varied factor per run."""
    return [
        # Baseline. Referenced by every comparison group below.
        Run("mnist_k2", "K=2 (128 total)", "clients"),
        # Party-count sweep A — FIXED per-party hidden_dim (64). The server
        # tail input is K*hidden_dim, so total cut width GROWS with K. This
        # mode is capacity-confounded: more parties also means a wider server
        # input. Use it to see "does the pipeline scale with K", not "do more
        # parties help".
        Run("mnist_k4", "K=4 (256 total)", "clients", clients=4),
        Run("mnist_k7", "K=7 (448 total)", "clients", clients=7),
        # Party-count sweep B — FIXED total cut width (~128). hidden_dim is
        # shrunk as K grows so K*hidden_dim stays roughly constant, isolating
        # the effect of party count from server-input width. K=2 anchor is the
        # baseline run above (K=2, hidden=64 → 128).
        Run(
            "mnist_ftw_k4",
            "K=4, hidden=32 (128 total)",
            "clients_ftw",
            clients=4,
            hidden_dim=32,
        ),
        Run(
            "mnist_ftw_k7",
            "K=7, hidden=18 (126 total)",
            "clients_ftw",
            clients=7,
            hidden_dim=18,
        ),
        # Learning-rate sweep
        Run("mnist_lr0.02", "K=2 lr=0.02", "lr", learning_rate=0.02),
        Run("mnist_lr0.1", "K=2 lr=0.1", "lr", learning_rate=0.1),
        # Cut-width sweep (hidden-dim)
        Run("mnist_h32", "K=2 hidden=32", "hidden", hidden_dim=32),
        Run("mnist_h128", "K=2 hidden=128", "hidden", hidden_dim=128),
    ]


def execute(run: Run, dry_run: bool) -> None:
    cmd = [
        "uv",
        "run",
        "flwr",
        "run",
        ".",
        run.federation,
        "--run-config",
        run.run_config(),
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
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands, do not run."
    )
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
        "num_server_rounds": NUM_SERVER_ROUNDS,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "batch_size": BATCH_SIZE,
        "eval_every": EVAL_EVERY,
        "eval_max_samples": EVAL_MAX_SAMPLES,
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
