"""Plot LoRA train loss from CSV, trainer_state.json, or a train .out log.

Examples:
    # Laptop (from checked-in artifacts):
    python -m scripts.training.plot_lora_training \\
        --csv training/artifacts/lora_v1/train_loss.csv \\
        --output-dir training/artifacts/lora_v1

    # Cluster:
    python -m scripts.training.plot_lora_training \\
        --trainer-state .../checkpoint-160/trainer_state.json \\
        --output-dir /mnt/home/dogakonuk/lora/eval/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _opt_float(value: str | None) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    return float(value)


def load_from_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("loss"):
                continue
            step_raw = row.get("step")
            rows.append(
                {
                    "step": int(float(step_raw)) if step_raw not in (None, "", "None") else None,
                    "epoch": _opt_float(row.get("epoch")),
                    "loss": float(row["loss"]),
                    "learning_rate": _opt_float(row.get("learning_rate")),
                    "mean_token_accuracy": _opt_float(row.get("mean_token_accuracy")),
                }
            )
    return rows


def load_from_trainer_state(path: Path) -> tuple[list[dict], dict]:
    st = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for h in st.get("log_history", []):
        if "loss" not in h:
            continue
        rows.append(
            {
                "step": h.get("step"),
                "epoch": h.get("epoch"),
                "loss": float(h["loss"]),
                "learning_rate": h.get("learning_rate"),
                "mean_token_accuracy": h.get("mean_token_accuracy"),
            }
        )
    meta = {
        "epoch": st.get("epoch"),
        "global_step": st.get("global_step"),
        "best_metric": st.get("best_metric"),
    }
    return rows, meta


def load_from_train_log(path: Path) -> list[dict]:
    """Parse Trainer dict lines printed to SLURM .out (fallback)."""
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    step = 0
    for m in re.finditer(r"\{[^{}]*'loss'\s*:\s*'?(?P<loss>[0-9.eE+-]+)'?[^{}]*\}", text):
        loss_m = re.search(r"'loss'\s*:\s*'?(?P<v>[0-9.eE+-]+)'?", m.group(0))
        ep_m = re.search(r"'epoch'\s*:\s*'?(?P<v>[0-9.eE+-]+)'?", m.group(0))
        if not loss_m:
            continue
        step += 5  # logging_steps default
        rows.append(
            {
                "step": step,
                "epoch": float(ep_m.group("v")) if ep_m else None,
                "loss": float(loss_m.group("v")),
                "learning_rate": None,
                "mean_token_accuracy": None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="train_loss.csv from artifacts")
    parser.add_argument("--trainer-state", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=None, help="Fallback: lora_train_*.out")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.csv and not args.trainer_state and not args.train_log:
        raise SystemExit("Provide --csv and/or --trainer-state and/or --train-log")

    meta: dict = {}
    if args.csv and args.csv.exists():
        rows = load_from_csv(args.csv)
        source = str(args.csv)
        summary_sidecar = args.csv.with_name("train_loss_summary.json")
        if summary_sidecar.exists():
            try:
                meta = json.loads(summary_sidecar.read_text(encoding="utf-8")).get(
                    "trainer_meta", {}
                )
            except json.JSONDecodeError:
                meta = {}
    elif args.trainer_state and args.trainer_state.exists():
        rows, meta = load_from_trainer_state(args.trainer_state)
        source = str(args.trainer_state)
    elif args.train_log and args.train_log.exists():
        rows = load_from_train_log(args.train_log)
        source = str(args.train_log)
    else:
        raise SystemExit("No readable --csv / trainer_state / train log")

    if not rows:
        raise SystemExit(f"No loss entries found in {source}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "train_loss.csv"
    # Don't clobber the input CSV if output-dir is the same folder.
    if args.csv is None or csv_path.resolve() != args.csv.resolve():
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "epoch",
                    "loss",
                    "learning_rate",
                    "mean_token_accuracy",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "source": source,
        "n_points": len(rows),
        "first_loss": rows[0]["loss"],
        "last_loss": rows[-1]["loss"],
        "min_loss": min(r["loss"] for r in rows),
        "trainer_meta": meta,
        "note": (
            "Train loss only — no eval_loss was logged (no val split / eval_strategy). "
            "Use held-out test node accuracy (base vs LoRA) as the overfit check."
        ),
    }
    (out / "train_loss_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless / no display
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"CSV ready at {csv_path} (matplotlib missing: {exc}; skip PNG)")
        print(json.dumps(summary, indent=2))
        return

    steps = [r["step"] if r.get("step") is not None else i for i, r in enumerate(rows, 1)]
    losses = [r["loss"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, losses, marker="o", markersize=3.5, linewidth=1.5, color="#1f4e79")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Train loss")
    ax.set_title("LoRA fine-tune — train loss (Qwen3-VL-8B)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out / "train_loss.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    print(f"Wrote {png}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
