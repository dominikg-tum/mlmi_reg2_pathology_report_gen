"""Batch thumbnail baselines over a split from chains.jsonl."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baselines.agent_runner import (
    default_runs_dir,
    load_paths_config,
    run_agent_traversal,
    write_phase1_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAINS = REPO_ROOT / "data" / "labels" / "chains.jsonl"


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    memory: str
    visual: str
    retriever: str


BASELINES: dict[str, BaselineSpec] = {
    "a": BaselineSpec("baseline_a_flat", memory="flat", visual="thumbnail", retriever="none"),
    "b1": BaselineSpec(
        "baseline_b1_hipporag2", memory="hipporag2", visual="thumbnail", retriever="none"
    ),
    "b2": BaselineSpec(
        "baseline_b2_hybridrag", memory="hybridrag", visual="thumbnail", retriever="none"
    ),
}


def load_slide_ids(chains_path: Path, *, split: str = "") -> list[str]:
    ids: list[str] = []
    with chains_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("extraction_status", "ok") != "ok":
                continue
            slide_id = str(raw.get("slide_id", "")).strip()
            if not slide_id:
                continue
            if split and raw.get("split") != split:
                continue
            ids.append(slide_id)
    return sorted(set(ids))


def default_runs_dir_for_baseline(baseline: BaselineSpec, cfg: dict | None = None) -> Path:
    work = Path((cfg or load_paths_config())["user"]["work_dir"])
    return work / "runs" / baseline.name


def run_one_slide(
    slide_id: str,
    *,
    baseline: BaselineSpec,
    runs_dir: Path,
    backend: str = "qwen",
    navigator: str = "graph_guided",
    skip_existing: bool = False,
) -> Path | None:
    out_path = runs_dir / slide_id / "cot_chain.json"
    if skip_existing and out_path.exists():
        print(f"Skip existing: {out_path}")
        return out_path

    result = run_agent_traversal(
        backend=backend,
        memory=baseline.memory,
        visual=baseline.visual,
        retriever=baseline.retriever,
        navigator=navigator,
        slide_id=slide_id,
        skip_report_nodes=False,
    )
    path = write_phase1_outputs(result, runs_dir, slide_id)
    print(f"Completed {slide_id} -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run thumbnail baselines (A / B1 / B2) over chains.jsonl split."
    )
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        required=True,
        help="a=flat, b1=hipporag2, b2=hybridrag",
    )
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--slide-id", type=str, default="", help="Run a single slide only")
    parser.add_argument(
        "--slide-index",
        type=int,
        default=-1,
        help="Run one slide by index into the split-filtered list (for SLURM arrays)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["dummy", "qwen"], default="qwen")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--write-slide-list",
        type=Path,
        default=None,
        help="Write filtered slide_ids (one per line) and exit",
    )
    parser.add_argument("--dry-run", action="store_true", help="List slides only")
    args = parser.parse_args()

    if not args.chains.exists():
        raise SystemExit(f"Missing chains file: {args.chains}")

    spec = BASELINES[args.baseline]
    slide_ids = load_slide_ids(args.chains, split=args.split)
    if args.slide_id:
        if args.slide_id not in slide_ids and args.split:
            raise SystemExit(
                f"Slide {args.slide_id!r} not in chains split={args.split!r}"
            )
        slide_ids = [args.slide_id]
    elif args.slide_index >= 0:
        if args.slide_index >= len(slide_ids):
            raise SystemExit(
                f"slide-index {args.slide_index} out of range (n={len(slide_ids)})"
            )
        slide_ids = [slide_ids[args.slide_index]]

    if args.write_slide_list:
        args.write_slide_list.parent.mkdir(parents=True, exist_ok=True)
        args.write_slide_list.write_text("\n".join(slide_ids) + ("\n" if slide_ids else ""))
        print(f"Wrote {len(slide_ids)} slide ids -> {args.write_slide_list}")

    if args.dry_run:
        print(f"baseline={args.baseline} split={args.split!r} slides={len(slide_ids)}")
        for slide_id in slide_ids[:20]:
            print(f"  {slide_id}")
        if len(slide_ids) > 20:
            print(f"  ... and {len(slide_ids) - 20} more")
        return

    if args.write_slide_list and not args.slide_id and args.slide_index < 0:
        return

    runs_dir = args.runs_dir or default_runs_dir_for_baseline(spec)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for slide_id in slide_ids:
        run_one_slide(
            slide_id,
            baseline=spec,
            runs_dir=runs_dir,
            backend=args.backend,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
