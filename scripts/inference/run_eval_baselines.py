"""Score all baseline run directories against chains.jsonl."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from baselines.agent_runner import load_paths_config
from scripts.inference.run_baseline_batch import BASELINES, default_runs_dir_for_baseline

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAINS = REPO_ROOT / "data" / "labels" / "chains.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build predictions.jsonl and run eval for baseline A/B1/B2."
    )
    parser.add_argument(
        "--baseline",
        choices=["all", *sorted(BASELINES)],
        default="all",
    )
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs root")
    args = parser.parse_args()

    if not args.chains.exists():
        raise SystemExit(f"Missing chains file: {args.chains}")

    cfg = load_paths_config()
    keys = sorted(BASELINES) if args.baseline == "all" else [args.baseline]

    for key in keys:
        spec = BASELINES[key]
        if args.runs_dir is not None:
            runs_dir = args.runs_dir / spec.name
        else:
            runs_dir = default_runs_dir_for_baseline(spec, cfg)
        pred_path = runs_dir / "predictions.jsonl"
        print(f"\n=== baseline {key} ({spec.name}) ===")
        print(f"runs_dir: {runs_dir}")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.inference.build_predictions",
                "--runs-dir",
                str(runs_dir),
                "--output",
                str(pred_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_eval",
                "--pred",
                str(pred_path),
                "--gt",
                str(args.chains),
                "--split",
                args.split,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
