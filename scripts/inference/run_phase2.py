"""Phase 2: cot_chain + slide embedding → MedGemma CAP report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.report_writer import (
    MedGemmaReportBackend,
    SlideProjector,
    load_cot_chain,
    write_report,
)
from baselines.agent_runner import default_runs_dir, load_paths_config
from eval.edge_parser import chain_dict_to_record, write_pred_edges
from scripts.vision._common import default_cache_root, load_vision_config
from vision.cache import build_slide_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 report generation with MedGemma.")
    parser.add_argument("--slide-id", required=True)
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
    slide_dir = runs_dir / args.slide_id

    chain_path = slide_dir / "cot_chain.json"
    if not chain_path.exists():
        raise SystemExit(f"Missing Phase 1 output: {chain_path}")

    chain = load_cot_chain(chain_path)
    slide_cache = build_slide_cache(cache_root, args.slide_id)
    slide_emb = slide_cache.load_slide_embedding() if slide_cache else None

    projector = SlideProjector()
    proj_path = args.projector_path or (slide_dir / "projector.pt")
    projector.load(proj_path)

    if args.dry_run:
        from agent.report_writer import build_report_prompt

        prefix = projector.project(slide_emb) if slide_emb is not None else None
        prompt = build_report_prompt(chain, slide_prefix=prefix)
        print(prompt)
        return

    model_path = str(
        args.model_path or cfg.get("models", {}).get("medgemma_4b", "")
    )
    if not model_path:
        raise SystemExit("medgemma model path not configured in configs/paths.yaml")

    backend = MedGemmaReportBackend(model_path)
    report = backend.generate_report(
        chain,
        slide_emb=slide_emb,
        projector=projector,
        max_new_tokens=args.max_new_tokens,
    )

    report_path = slide_dir / "report.txt"
    write_report(report_path, report)
    projector.save(proj_path)

    record = chain_dict_to_record(chain, report=report)
    write_pred_edges(record, slide_dir / "pred_edges.jsonl")

    print(f"Phase 2 complete: {report_path}")
    print(json.dumps({"slide_id": args.slide_id, "report_len": len(report)}, indent=2))


if __name__ == "__main__":
    main()
