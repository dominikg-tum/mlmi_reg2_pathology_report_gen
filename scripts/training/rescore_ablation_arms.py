"""Rescore a/p0 × base/LoRA arms from existing predictions.jsonl (no regeneration).

Example (cluster)::

    python -m scripts.training.rescore_ablation_arms \\
      --runs-root /mnt/projects/mlmi/reg2/dogakonuk/runs \\
      --gt /mnt/projects/mlmi/reg2/dogakonuk/repos/mlmi_reg2_pathology_report_gen/data/labels/chains.jsonl \\
      --skip-bert --plot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ARMS = (
    ("a_base", "baseline_a_flat"),
    ("a_lora", "baseline_a_flat_lora"),
    ("p0_base", "baseline_p0_patch_cosine"),
    ("p0_lora", "baseline_p0_patch_cosine_lora"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore ablation arms + optional plots")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument(
        "--include-report-node",
        action="store_true",
        help="Legacy scoring that keeps report in CoT path metrics",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="metrics.json",
        help="Overwrite this file inside each arm dir",
    )
    parser.add_argument("--plot", action="store_true", help="Also write CoT/Report bar charts")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output dir (default: <runs-root>/plots)",
    )
    args = parser.parse_args()

    summary = []
    for label, folder in DEFAULT_ARMS:
        runs_dir = args.runs_root / folder
        pred = runs_dir / "predictions.jsonl"
        if not pred.exists():
            raise SystemExit(f"Missing {pred}")
        out = runs_dir / args.metrics_name
        cmd = [
            sys.executable,
            "-m",
            "eval.run_eval",
            "--pred",
            str(pred),
            "--gt",
            str(args.gt),
            "--split",
            args.split,
            "--json-out",
            str(out),
        ]
        if args.skip_bert:
            cmd.append("--skip-bert")
        if args.include_report_node:
            cmd.append("--include-report-node")
        print(f"\n=== {label} ({folder}) ===")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        metrics = json.loads(out.read_text())
        summary.append(
            {
                "arm": label,
                "edge_f1": metrics.get("edge_f1"),
                "node_accuracy": metrics.get("node_accuracy"),
                "binary_path_validity": metrics.get("binary_path_validity"),
                "final_diagnosis_accuracy": metrics.get("final_diagnosis_accuracy"),
                "rouge_l": metrics.get("rouge_l"),
                "mess": metrics.get("mess"),
            }
        )

    table_path = (args.out_dir or (args.runs_root / "plots"))
    table_path.mkdir(parents=True, exist_ok=True)
    summary_path = table_path / "ablation_rescore_table.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {summary_path}")
    print(
        f"{'arm':12} {'BPV':>7} {'EdgeF1':>7} {'NodeAcc':>7} {'Diag':>7} {'ROUGE':>7}"
    )
    for r in summary:
        d = r["final_diagnosis_accuracy"]
        d_s = f"{d:.3f}" if d is not None else "  N/A"
        print(
            f"{r['arm']:12} {r['binary_path_validity']:7.3f} "
            f"{r['edge_f1']:7.3f} {r['node_accuracy']:7.3f} "
            f"{d_s:>7} {r['rouge_l']:7.3f}"
        )

    if args.plot:
        plot_cmd = [
            sys.executable,
            "-m",
            "scripts.training.plot_ablation_metrics",
            "--runs-root",
            str(args.runs_root),
            "--metrics-name",
            args.metrics_name,
            "--out-dir",
            str(args.out_dir or (args.runs_root / "plots")),
        ]
        subprocess.run(plot_cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
