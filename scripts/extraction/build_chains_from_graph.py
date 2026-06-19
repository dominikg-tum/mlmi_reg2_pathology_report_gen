"""WP3: batch graph-walk CoT extraction from english_reports → chains.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from extraction.graph_walk import GraphWalkError, steps_to_chain_dict, walk_graph
from extraction.labels_io import (
    load_existing_slide_ids,
    load_failed_slide_ids,
    load_slides_from_xlsx,
    upsert_chains_jsonl,
    write_chains_jsonl,
)
from extraction.qa_extractor import load_config
from graph import load_graph

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "labels" / "chains.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract graph-aligned ground-truth chains from pathology reports."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xlsx", type=Path, default=None, help="Override labels xlsx path")
    parser.add_argument("--limit", type=int, default=0, help="Process only N slides (0=all)")
    parser.add_argument("--slide", type=str, default="", help="Single slide_id e.g. TUM_Uterus_0001.svs")
    parser.add_argument("--resume", action="store_true", help="Skip slides already in output")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-extract only failed slides and replace their records",
    )
    parser.add_argument("--dry-run", action="store_true", help="List slides only, no LLM calls")
    args = parser.parse_args()

    cfg = load_config()
    xlsx_path = args.xlsx or Path(cfg["cluster"]["labels_xlsx"])
    slides = load_slides_from_xlsx(
        xlsx_path, slide_filter=args.slide, limit=args.limit
    )

    if args.resume:
        done = load_existing_slide_ids(args.output)
        slides = [s for s in slides if s.slide_id not in done]
    elif args.retry_failed:
        failed = load_failed_slide_ids(args.output)
        slides = [s for s in slides if s.slide_id in failed]
        if not slides:
            print(f"No failed slides to retry in {args.output}")
            return

    print(f"Slides to process: {len(slides)} -> {args.output}")
    if args.dry_run:
        for s in slides[:20]:
            print(f"  {s.slide_id} ({s.split}) report_len={len(s.report)}")
        if len(slides) > 20:
            print(f"  ... and {len(slides) - 20} more")
        return

    from extraction.qa_extractor import build_client

    graph, root_id = load_graph()
    client = build_client(cfg)
    model = cfg["qwen"]["model_name"]

    ok, failed = 0, 0
    batch: list[dict] = []
    file_exists = args.output.exists() and args.resume and not args.retry_failed

    for slide in tqdm(slides, desc="extract chains"):
        if not slide.report:
            record = steps_to_chain_dict(
                slide.slide_id,
                [],
                "",
                slide.split,
                extraction_status="failed",
                error="empty report",
            )
            batch.append(record)
            failed += 1
            continue

        try:
            result = walk_graph(slide.report, graph, root_id, client, model)
            record = steps_to_chain_dict(
                slide.slide_id,
                result.steps,
                slide.report,
                slide.split,
            )
            batch.append(record)
            ok += 1
        except GraphWalkError as exc:
            record = steps_to_chain_dict(
                slide.slide_id,
                [],
                slide.report,
                slide.split,
                extraction_status="failed",
                error=str(exc),
            )
            batch.append(record)
            failed += 1

        if len(batch) >= 10:
            if args.retry_failed:
                upsert_chains_jsonl(batch, args.output)
            else:
                write_chains_jsonl(batch, args.output, append=file_exists)
                file_exists = True
            batch.clear()

    if batch:
        if args.retry_failed:
            upsert_chains_jsonl(batch, args.output)
        else:
            write_chains_jsonl(batch, args.output, append=file_exists)

    print(f"Done: ok={ok} failed={failed} -> {args.output}")


if __name__ == "__main__":
    main()
