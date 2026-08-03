"""Bar charts: CoT vs Report metrics for a/p0 × base/LoRA ablation arms.

Layout (team preference):
  - x-axis = metrics (BPV, Edge-F1, …)
  - colors / legend = arms (a_base, a_lora, p0_base, p0_lora)
  - CoT and Report stay in separate figures

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

# Distinct colors per arm (stable legend order)
ARM_COLORS = {
    "a_base": "#1B4F72",   # navy
    "a_lora": "#2E86C1",   # blue
    "p0_base": "#B9770E",  # amber
    "p0_lora": "#1D8348",  # green
}

COT_KEYS = (
    ("binary_path_validity", "Binary Path Val."),
    ("final_diagnosis_accuracy", "Final Diag. Acc"),
    ("edge_f1", "Edge-F1"),
    ("mess", "MESS"),
    ("node_accuracy", "Node Accuracy"),
)

REPORT_KEYS = (
    ("rouge_l", "ROUGE-L"),
    ("bleu4", "BLEU-4"),
    ("clinical_token_f1", "Clinical (proxy)"),
    ("numeric_fidelity", "Num. FID"),
    ("negation_consistency", "Negation Cons."),
    # BERTScore omitted unless present in metrics.json (often skipped for speed).
    ("bert_score_f1", "BERT"),
)


def _load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cot = data.get("cot") or {}
    rep = data.get("report") or {}
    flat = {
        **{k: data.get(k) for k, _ in COT_KEYS + REPORT_KEYS},
        **cot,
        **rep,
    }
    return flat


def _bar_by_metric(
    ax,
    metric_labels: list[str],
    arm_series: list[tuple[str, list[float]]],
    title: str,
) -> None:
    """Grouped bars: one group per metric on x; one color per arm."""
    n_metrics = len(metric_labels)
    n_arms = len(arm_series)
    width = 0.8 / max(n_arms, 1)
    x = list(range(n_metrics))
    for i, (arm, values) in enumerate(arm_series):
        offsets = [xi + (i - (n_arms - 1) / 2) * width for xi in x]
        ax.bar(
            offsets,
            values,
            width=width,
            label=arm,
            color=ARM_COLORS.get(arm),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="variant", fontsize=9, loc="upper right")
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

    rows: list[dict] = []
    for label, folder in DEFAULT_ARMS:
        path = args.runs_root / folder / args.metrics_name
        if not path.exists():
            alt = args.runs_root / f"metrics_{label}.json"
            if alt.exists():
                path = alt
            else:
                raise SystemExit(f"Missing metrics for {label}: {path}")
        m = _load_metrics(path)
        rows.append({"arm": label, **{k: m.get(k) for k, _ in COT_KEYS + REPORT_KEYS}})

    # CoT: x = metrics, colors = arms
    cot_labels = [pretty for _, pretty in COT_KEYS]
    cot_series = []
    for r in rows:
        vals = []
        for key, _ in COT_KEYS:
            v = r.get(key)
            vals.append(float(v) if v is not None else 0.0)
        cot_series.append((r["arm"], vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_by_metric(
        ax,
        cot_labels,
        cot_series,
        "Chain-of-Thought and Graph Metrics",
    )
    fig.tight_layout()
    cot_png = out_dir / "ablation_cot_metrics.png"
    fig.savefig(cot_png, dpi=160)
    plt.close(fig)

    # Report: same layout. Drop metrics that are missing for every arm (e.g. BERT skipped).
    active_report_keys = []
    for key, pretty in REPORT_KEYS:
        if any(r.get(key) is not None for r in rows):
            active_report_keys.append((key, pretty))
    if not active_report_keys:
        active_report_keys = list(REPORT_KEYS[:3])

    rep_labels = [pretty for _, pretty in active_report_keys]
    rep_series = []
    for r in rows:
        vals = []
        for key, _ in active_report_keys:
            v = r.get(key)
            vals.append(float(v) if v is not None else 0.0)
        rep_series.append((r["arm"], vals))

    fig, ax = plt.subplots(figsize=(11, 5))
    _bar_by_metric(ax, rep_labels, rep_series, "Report Generation Metrics")
    fig.tight_layout()
    rep_png = out_dir / "ablation_report_metrics.png"
    fig.savefig(rep_png, dpi=160)
    plt.close(fig)

    csv_path = out_dir / "ablation_metrics_summary.csv"
    fieldnames = ["arm"] + [k for k, _ in COT_KEYS + REPORT_KEYS]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    json_path = out_dir / "ablation_metrics_summary.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {cot_png}")
    print(f"Wrote {rep_png}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
