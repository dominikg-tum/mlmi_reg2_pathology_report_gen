"""Bar charts: CoT vs Report metrics for a/p0 × base/LoRA ablation arms.

Reads metrics.json files produced by ``python -m eval.run_eval --json-out ...``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_ARMS = (
    ("a_base", "baseline_a_flat"),
    ("a_lora", "baseline_a_flat_lora"),
    ("p0_base", "baseline_p0_patch_cosine"),
    ("p0_lora", "baseline_p0_patch_cosine_lora"),
)

COT_KEYS = (
    ("binary_path_validity", "BPV"),
    ("edge_f1", "Edge-F1"),
    ("node_accuracy", "Node Acc"),
    ("mess", "MESS"),
    ("diagnosis_label_accuracy", "Diag Acc"),
)

REPORT_KEYS = (
    ("rouge_l", "ROUGE-L"),
    ("bleu4", "BLEU-4"),
    ("clinical_token_f1", "Clin F1"),
)


def _load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    # Prefer nested sections when present
    cot = data.get("cot") or {}
    rep = data.get("report") or {}
    flat = {
        **{k: data.get(k) for k, _ in COT_KEYS + REPORT_KEYS},
        **cot,
        **rep,
    }
    return flat


def _bar_group(
    ax,
    arms: list[str],
    series: list[tuple[str, list[float]]],
    title: str,
) -> None:
    n_arms = len(arms)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    x = list(range(n_arms))
    for i, (label, values) in enumerate(series):
        offsets = [xi + (i - (n_series - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CoT vs Report ablation metrics")
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Directory containing baseline_* run folders with metrics.json",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="metrics.json",
        help="Filename inside each arm dir (default metrics.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <runs-root>/plots)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.runs_root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_labels: list[str] = []
    rows: list[dict] = []
    for label, folder in DEFAULT_ARMS:
        path = args.runs_root / folder / args.metrics_name
        if not path.exists():
            # Also allow flat layout: runs-root/metrics_<label>.json
            alt = args.runs_root / f"metrics_{label}.json"
            if alt.exists():
                path = alt
            else:
                raise SystemExit(f"Missing metrics for {label}: {path}")
        m = _load_metrics(path)
        arm_labels.append(label)
        row = {"arm": label, **{k: m.get(k) for k, _ in COT_KEYS + REPORT_KEYS}}
        rows.append(row)

    # CoT figure
    fig, ax = plt.subplots(figsize=(9, 4.5))
    series = []
    for key, pretty in COT_KEYS:
        vals = []
        for r in rows:
            v = r.get(key)
            vals.append(float(v) if v is not None else 0.0)
        series.append((pretty, vals))
    _bar_group(ax, arm_labels, series, "CoT metrics (report node excluded)")
    fig.tight_layout()
    cot_png = out_dir / "ablation_cot_metrics.png"
    fig.savefig(cot_png, dpi=160)
    plt.close(fig)

    # Report figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = []
    for key, pretty in REPORT_KEYS:
        vals = [float(r.get(key) or 0.0) for r in rows]
        series.append((pretty, vals))
    _bar_group(ax, arm_labels, series, "Report metrics")
    fig.tight_layout()
    rep_png = out_dir / "ablation_report_metrics.png"
    fig.savefig(rep_png, dpi=160)
    plt.close(fig)

    # Summary CSV + JSON
    csv_path = out_dir / "ablation_metrics_summary.csv"
    fieldnames = ["arm"] + [k for k, _ in COT_KEYS + REPORT_KEYS]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    json_path = out_dir / "ablation_metrics_summary.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"Wrote {cot_png}")
    print(f"Wrote {rep_png}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
