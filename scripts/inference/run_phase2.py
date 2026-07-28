"""Phase 2: SS-LLM merge of per-slide chains → MedGemma CAP report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent.report_writer import (
    MedGemmaReportBackend,
    SlideProjector,
    build_merged_report_prompt,
    load_cot_chain,
    mean_slide_prefix,
    merge_case_chains,
    write_merged_case_chain,
    write_report,
)
from baselines.agent_runner import default_runs_dir, load_paths_config
from eval.edge_parser import chain_dict_to_record, write_pred_edges
from extraction.case_ids import case_run_dir, case_spec_from_key, physical_run_dir
from scripts.vision._common import default_cache_root, load_vision_config
from vision.cache import build_slide_cache


def _load_per_slide_chains(
    runs_dir: Path, case_key: str, physical_slides: list[str]
) -> tuple[list[dict], list[str]]:
    """Load per-slide chains; fall back to case-level merged chain if needed."""
    chains: list[dict] = []
    missing: list[str] = []
    for physical_id in physical_slides:
        path = physical_run_dir(runs_dir, case_key, physical_id) / "cot_chain.json"
        if path.exists():
            chains.append(load_cot_chain(path))
        else:
            missing.append(physical_id)

    if not missing:
        return chains, physical_slides

    case_chain_path = case_run_dir(runs_dir, case_key) / "cot_chain.json"
    if case_chain_path.exists() and len(chains) == 0:
        # Legacy / skipped layout: only case-level chain available.
        case_chain = load_cot_chain(case_chain_path)
        return [case_chain], [case_key]

    if missing:
        raise SystemExit(
            "Missing per-slide Phase 1 outputs for "
            f"{missing}; expected under "
            f"{physical_run_dir(runs_dir, case_key, missing[0]).parent}/"
            f" or a complete case-level {case_chain_path}"
        )
    return chains, physical_slides


def _load_slide_embs(
    cache_root: Path | None, physical_slides: list[str]
) -> list[np.ndarray | None]:
    embs: list[np.ndarray | None] = []
    for physical_id in physical_slides:
        slide_cache = (
            build_slide_cache(cache_root, physical_id) if cache_root else None
        )
        emb = slide_cache.load_slide_embedding() if slide_cache else None
        embs.append(emb)
    return embs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 SS-LLM report generation with MedGemma."
    )
    parser.add_argument(
        "--slide-id",
        required=True,
        help="Case key / GT slide_id (comma-separated for multi-WSI)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--projector-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Build prompt only, no GPU load")
    args = parser.parse_args()

    cfg = load_paths_config()
    vcfg = load_vision_config()
    runs_dir = args.runs_dir or default_runs_dir(cfg)
    cache_root = args.cache_root or default_cache_root(vcfg)

    case = case_spec_from_key(args.slide_id)
    case_dir = case_run_dir(runs_dir, case.case_key)

    chains, chain_slide_ids = _load_per_slide_chains(
        runs_dir, case.case_key, case.physical_slides
    )
    # When falling back to a single case-level chain, keep case_key as slide label.
    if chain_slide_ids == [case.case_key]:
        merged = dict(chains[0])
        merged["slide_id"] = case.case_key
        merged.setdefault("physical_slides", case.physical_slides)
        merged.setdefault("fusion", "ss_llm")
    else:
        merged = merge_case_chains(chains, chain_slide_ids, case.case_key)
    write_merged_case_chain(case_dir / "cot_chain.json", merged)

    slide_embs = _load_slide_embs(cache_root, case.physical_slides)
    projector = SlideProjector()
    proj_path = args.projector_path or (case_dir / "projector.pt")
    projector.load(proj_path)

    if args.dry_run:
        prefix = mean_slide_prefix(slide_embs, projector)
        prompt = build_merged_report_prompt(
            chains, slide_ids=chain_slide_ids, slide_prefix=prefix
        )
        print(prompt)
        return

    model_path = str(
        args.model_path or cfg.get("models", {}).get("medgemma_4b", "")
    )
    if not model_path:
        raise SystemExit("medgemma model path not configured in configs/paths.yaml")

    backend = MedGemmaReportBackend(model_path)
    report = backend.generate_report(
        chains=chains,
        slide_ids=chain_slide_ids,
        slide_embs=slide_embs,
        projector=projector,
        max_new_tokens=args.max_new_tokens,
    )

    report_path = case_dir / "report.txt"
    write_report(report_path, report)
    projector.save(proj_path)

    merged["report"] = report
    write_merged_case_chain(case_dir / "cot_chain.json", merged)

    record = chain_dict_to_record(merged, report=report)
    write_pred_edges(record, case_dir / "pred_edges.jsonl")

    print(f"Phase 2 complete: {report_path}")
    print(
        json.dumps(
            {
                "slide_id": case.case_key,
                "physical_slides": case.physical_slides,
                "report_len": len(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
