"""Aggregate runs/*/cot_chain.json → runs/predictions.jsonl for eval (case-level)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.agent_runner import default_runs_dir
from eval.edge_parser import parse_slide_run, record_to_eval_dict


def iter_case_dirs(runs_dir: Path) -> list[Path]:
    """Case dirs have cot_chain.json at the root (ignore nested slides/)."""
    return sorted(
        p
        for p in runs_dir.iterdir()
        if p.is_dir() and (p / "cot_chain.json").exists() and p.name != "slides"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build predictions.jsonl from per-case SS-LLM runs."
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs_dir = args.runs_dir or default_runs_dir()
    out_path = args.output or (runs_dir / "predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    case_dirs = iter_case_dirs(runs_dir)
    if not case_dirs:
        raise SystemExit(f"No case runs under {runs_dir}")

    lines: list[str] = []
    for case_dir in case_dirs:
        case_key = case_dir.name
        try:
            record = parse_slide_run(runs_dir, case_key)
        except FileNotFoundError:
            continue
        # Ensure eval join key matches GT (directory name = case key).
        if not record.slide_id:
            record.slide_id = case_key
        elif record.slide_id != case_key:
            record.slide_id = case_key
        lines.append(json.dumps(record_to_eval_dict(record)))

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {len(lines)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
