"""Batch baselines over a split from chains.jsonl (SS-LLM case unit)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from baselines.agent_runner import load_paths_config, run_case_phase1
from baselines.direct_report import run_naive_case
from extraction.case_ids import CaseSpec, load_cases_from_chains

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAINS = REPO_ROOT / "data" / "labels" / "chains.jsonl"


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    memory: str
    visual: str
    retriever: str
    node_react: bool = False
    structured_answer: bool = False
    paired_regions: bool = False
    mode: str = "graph"  # "graph" | "naive"


BASELINES: dict[str, BaselineSpec] = {
    "a": BaselineSpec("baseline_a_flat", memory="flat", visual="thumbnail", retriever="none"),
    "b1": BaselineSpec(
        "baseline_b1_hipporag2", memory="hipporag2", visual="thumbnail", retriever="none"
    ),
    "b2": BaselineSpec(
        "baseline_b2_hybridrag", memory="hybridrag", visual="thumbnail", retriever="none"
    ),
    "b2_cap": BaselineSpec(
        "baseline_b2_hybridrag_cap",
        memory="hybridrag_cap",
        visual="thumbnail",
        retriever="none",
    ),
    "p0": BaselineSpec(
        "baseline_p0_patch_cosine", memory="flat", visual="patch_retrieve", retriever="graph_guided"
    ),
    "p1": BaselineSpec(
        "baseline_p1_patch_structured",
        memory="flat",
        visual="patch_retrieve",
        retriever="graph_guided",
        structured_answer=True,
    ),
    "p2": BaselineSpec(
        "baseline_p2_patch_node_react",
        memory="flat",
        visual="patch_retrieve",
        retriever="graph_guided",
        node_react=True,
    ),
    "p3": BaselineSpec(
        "baseline_p3_patch_node_react_paired",
        memory="flat",
        visual="patch_retrieve",
        retriever="graph_guided",
        node_react=True,
        paired_regions=True,
    ),
    "naive": BaselineSpec(
        "baseline_naive_oneshot",
        memory="flat",
        visual="thumbnail",
        retriever="none",
        mode="naive",
    ),
}


def load_slide_ids(chains_path: Path, *, split: str = "") -> list[str]:
    """Backward-compatible: return sorted case keys (GT slide_id strings)."""
    return [c.case_key for c in load_cases_from_chains(chains_path, split=split)]


def default_runs_dir_for_baseline(baseline: BaselineSpec, cfg: dict | None = None) -> Path:
    work = Path((cfg or load_paths_config())["user"]["work_dir"])
    return work / "runs" / baseline.name


def run_one_case(
    case: CaseSpec,
    *,
    baseline: BaselineSpec,
    runs_dir: Path,
    backend: str = "qwen",
    navigator: str = "graph_guided",
    skip_existing: bool = False,
) -> Path | None:
    if baseline.mode == "naive":
        path = run_naive_case(
            case,
            runs_dir=runs_dir,
            backend=backend,
            skip_existing=skip_existing,
        )
    else:
        path = run_case_phase1(
            case,
            runs_dir=runs_dir,
            backend=backend,
            memory=baseline.memory,
            visual=baseline.visual,
            retriever=baseline.retriever,
            navigator=navigator,
            skip_report_nodes=False,
            node_react=baseline.node_react,
            structured_answer=baseline.structured_answer,
            paired_regions=baseline.paired_regions,
            skip_existing=skip_existing,
        )
    print(f"Completed case {case.case_key} ({len(case.physical_slides)} slides) -> {path}")
    return path


def run_one_slide(
    slide_id: str,
    *,
    baseline: BaselineSpec,
    runs_dir: Path,
    backend: str = "qwen",
    navigator: str = "graph_guided",
    skip_existing: bool = False,
) -> Path | None:
    """Backward-compatible alias: treat slide_id as a case key."""
    from extraction.case_ids import case_spec_from_key

    return run_one_case(
        case_spec_from_key(slide_id),
        baseline=baseline,
        runs_dir=runs_dir,
        backend=backend,
        navigator=navigator,
        skip_existing=skip_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baselines over chains.jsonl split (SS-LLM: one case = all WSIs)."
    )
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        required=True,
        help=(
            "a/b1/b2/b2_cap=thumbnail graph; p0-p3=patch; naive=thumbnail one-shot no graph. "
            "b2=HybridRAG reports-only; b2_cap=HybridRAG+CAP refs. "
            "All use SS-LLM Pick (per-slide Phase 1, one selected case chain)."
        ),
    )
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--slide-id",
        type=str,
        default="",
        help="Run a single case (GT slide_id key; may be comma-separated)",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="",
        help="Alias for --slide-id (case key)",
    )
    parser.add_argument(
        "--slide-index",
        type=int,
        default=-1,
        help="Run one case by index into the split-filtered case list (for SLURM arrays)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["dummy", "qwen", "finetuned"], default="qwen")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--write-slide-list",
        type=Path,
        default=None,
        help="Write filtered case keys (one per line) and exit",
    )
    parser.add_argument("--dry-run", action="store_true", help="List cases only")
    args = parser.parse_args()

    if not args.chains.exists():
        raise SystemExit(f"Missing chains file: {args.chains}")

    spec = BASELINES[args.baseline]
    cases = load_cases_from_chains(args.chains, split=args.split)
    case_key = args.case_id or args.slide_id
    if case_key:
        keys = {c.case_key for c in cases}
        if case_key not in keys and args.split:
            raise SystemExit(
                f"Case {case_key!r} not in chains split={args.split!r}"
            )
        from extraction.case_ids import case_spec_from_key

        cases = [case_spec_from_key(case_key, split=args.split)]
    elif args.slide_index >= 0:
        if args.slide_index >= len(cases):
            raise SystemExit(
                f"slide-index {args.slide_index} out of range (n_cases={len(cases)})"
            )
        cases = [cases[args.slide_index]]

    case_keys = [c.case_key for c in cases]

    if args.write_slide_list:
        args.write_slide_list.parent.mkdir(parents=True, exist_ok=True)
        args.write_slide_list.write_text(
            "\n".join(case_keys) + ("\n" if case_keys else "")
        )
        print(f"Wrote {len(case_keys)} case keys -> {args.write_slide_list}")

    if args.dry_run:
        n_slides = sum(len(c.physical_slides) for c in cases)
        print(
            f"baseline={args.baseline} split={args.split!r} "
            f"cases={len(cases)} physical_slides={n_slides}"
        )
        for case in cases[:20]:
            print(f"  {case.case_key}  (n={len(case.physical_slides)})")
        if len(cases) > 20:
            print(f"  ... and {len(cases) - 20} more")
        return

    if args.write_slide_list and not case_key and args.slide_index < 0:
        return

    runs_dir = args.runs_dir or default_runs_dir_for_baseline(spec)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        run_one_case(
            case,
            baseline=spec,
            runs_dir=runs_dir,
            backend=args.backend,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
