"""Aggregate runs/*/pred_edges.jsonl → runs/predictions.jsonl for eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.agent_runner import default_runs_dir
from eval.edge_parser import parse_slide_run, record_to_eval_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predictions.jsonl from per-slide runs.")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs_dir = args.runs_dir or default_runs_dir()
    out_path = args.output or (runs_dir / "predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slide_dirs = sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and (p / "cot_chain.json").exists()
    )
    if not slide_dirs:
        raise SystemExit(f"No slide runs under {runs_dir}")

    lines: list[str] = []
    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        try:
            record = parse_slide_run(runs_dir, slide_id)
        except FileNotFoundError:
            continue
        lines.append(json.dumps(record_to_eval_dict(record)))

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {len(lines)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
