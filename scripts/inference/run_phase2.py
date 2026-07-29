"""Phase 2: selected SS-LLM chain → MedGemma CAP report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.report_writer import (
    MedGemmaReportBackend,
    SlideProjector,
    build_report_prompt,
    load_cot_chain,
    write_case_chain,
    write_report,
)
from agent.slide_selector import (
    build_selected_case_chain,
    select_slide_chain,
    selection_from_case_chain,
    write_case_meta,
)
from baselines.agent_runner import (
    build_selector_backend,
    default_runs_dir,
    load_paths_config,
)
from eval.edge_parser import chain_dict_to_record, write_pred_edges
from extraction.case_ids import case_run_dir, case_spec_from_key, physical_run_dir
from scripts.vision._common import default_cache_root, load_vision_config
from vision.cache import build_slide_cache


def _load_per_slide_chains(
    runs_dir: Path, case_key: str, physical_slides: list[str]
) -> list[dict]:
    chains: list[dict] = []
    missing: list[str] = []
    for physical_id in physical_slides:
        path = physical_run_dir(runs_dir, case_key, physical_id) / "cot_chain.json"
        if path.exists():
            chains.append(load_cot_chain(path))
        else:
            missing.append(physical_id)

    if missing:
        raise SystemExit(
            "Missing per-slide Phase 1 outputs for "
            f"{missing}; expected under "
            f"{physical_run_dir(runs_dir, case_key, missing[0]).parent}/"
            ". Rerun Phase 1 to create the SS-LLM Pick layout."
        )
    return chains


def _load_selected_case_chain(
    runs_dir: Path,
    case_key: str,
    physical_slides: list[str],
    *,
    selector_backend: Any | None = None,
) -> tuple[dict, str]:
    """Load the Phase 1 pick, migrating legacy case outputs with the same selector."""
    case_dir = case_run_dir(runs_dir, case_key)
    case_chain_path = case_dir / "cot_chain.json"
    if case_chain_path.exists():
        case_chain = load_cot_chain(case_chain_path)
        stored = selection_from_case_chain(case_chain, physical_slides)
        if stored is not None:
            meta_path = case_dir / "case_meta.json"
            if not meta_path.exists():
                write_case_meta(
                    meta_path,
                    case_key=case_key,
                    physical_slides=physical_slides,
                    selection=stored,
                )
            return case_chain, stored.chosen_slide_id

    chains = _load_per_slide_chains(runs_dir, case_key, physical_slides)
    selection = select_slide_chain(
        chains, physical_slides, backend=selector_backend
    )
    selected_index = physical_slides.index(selection.chosen_slide_id)
    case_chain = build_selected_case_chain(
        chains[selected_index],
        case_key=case_key,
        physical_slides=physical_slides,
        selection=selection,
    )
    write_case_chain(case_chain_path, case_chain)
    write_case_meta(
        case_dir / "case_meta.json",
        case_key=case_key,
        physical_slides=physical_slides,
        selection=selection,
    )
    return case_chain, selection.chosen_slide_id


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
    parser.add_argument(
        "--selector-backend",
        choices=["qwen", "dummy", "finetuned"],
        default="qwen",
        help="Model used only when a legacy case run still needs an SS-LLM pick",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build prompt only, no GPU load")
    args = parser.parse_args()

    cfg = load_paths_config()
    vcfg = load_vision_config()
    runs_dir = args.runs_dir or default_runs_dir(cfg)
    cache_root = args.cache_root or default_cache_root(vcfg)

    case = case_spec_from_key(args.slide_id)
    case_dir = case_run_dir(runs_dir, case.case_key)

    selected_chain, selected_slide_id = _load_selected_case_chain(
        runs_dir,
        case.case_key,
        case.physical_slides,
        selector_backend=build_selector_backend(args.selector_backend),
    )
    selected_cache = (
        build_slide_cache(cache_root, selected_slide_id) if cache_root else None
    )
    slide_emb = selected_cache.load_slide_embedding() if selected_cache else None
    projector = SlideProjector()
    proj_path = args.projector_path or (case_dir / "projector.pt")
    projector.load(proj_path)

    if args.dry_run:
        prefix = projector.project(slide_emb) if slide_emb is not None else None
        prompt = build_report_prompt(selected_chain, slide_prefix=prefix)
        print(prompt)
        return

    model_path = str(
        args.model_path or cfg.get("models", {}).get("medgemma_4b", "")
    )
    if not model_path:
        raise SystemExit("medgemma model path not configured in configs/paths.yaml")

    backend = MedGemmaReportBackend(model_path)
    report = backend.generate_report(
        chain=selected_chain,
        slide_emb=slide_emb,
        projector=projector,
        max_new_tokens=args.max_new_tokens,
    )

    report_path = case_dir / "report.txt"
    write_report(report_path, report)
    projector.save(proj_path)

    selected_chain["report"] = report
    write_case_chain(case_dir / "cot_chain.json", selected_chain)

    record = chain_dict_to_record(selected_chain, report=report)
    write_pred_edges(record, case_dir / "pred_edges.jsonl")

    print(f"Phase 2 complete: {report_path}")
    print(
        json.dumps(
            {
                "slide_id": case.case_key,
                "physical_slides": case.physical_slides,
                "selected_slide_id": selected_slide_id,
                "report_len": len(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
