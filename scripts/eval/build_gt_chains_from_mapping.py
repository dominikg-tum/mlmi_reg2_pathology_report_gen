#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from extraction.graph_walk import GraphWalkError, steps_to_chain_dict, walk_graph
from graph import load_graph


REPORT_COLUMNS = [
    "Ori_WSI_level_report_en",
    "diagnostic_findings",
    "microscopic_description",
]


def load_predictions(path: Path) -> list[str]:
    slide_ids: list[str] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                slide_ids.append(json.loads(line)["slide_id"])
    return slide_ids


def pred_to_mapping_key(slide_id: str) -> str:
    match = re.search(r"TUM_Uterus_s(\d{4})_", slide_id)
    if not match:
        raise ValueError(f"Cannot parse prediction slide_id: {slide_id}")
    return f"TUM_Uterus_{match.group(1)}"


def best_report(row: pd.Series) -> str:
    for col in REPORT_COLUMNS:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line).get("slide_id", ""))
                except Exception:
                    pass
    return done


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-xlsx", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="/mnt/research/ljs/Qwen3-VL-8B-Instruct")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--sheet", default="seperatable_reports")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    pred_slide_ids = load_predictions(args.pred)
    if args.limit:
        pred_slide_ids = pred_slide_ids[: args.limit]

    df = pd.read_excel(args.mapping_xlsx, sheet_name=args.sheet)
    by_key = {
        str(row["anoy_img_id_new"]).replace(".svs", ""): row
        for _, row in df.iterrows()
        if pd.notna(row.get("anoy_img_id_new"))
    }

    done = load_done(args.output) if args.resume else set()
    if args.output.exists() and not args.resume:
        args.output.unlink()

    graph, root_id = load_graph()
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    ok = 0
    failed = 0
    pending = [sid for sid in pred_slide_ids if sid not in done]
    for slide_id in tqdm(pending, desc="GT graph chains", unit="slide", dynamic_ncols=True):
        try:
            key = pred_to_mapping_key(slide_id)
            if key not in by_key:
                raise KeyError(f"No mapping row for {slide_id} -> {key}")
            report = best_report(by_key[key])
            if not report:
                raise ValueError(f"Empty report for {slide_id}")

            last_error = ""
            for attempt in range(1, args.retries + 1):
                try:
                    result = walk_graph(report, graph, root_id, client, args.model)
                    record = steps_to_chain_dict(
                        slide_id,
                        result.steps,
                        report,
                        "test",
                    )
                    append_record(args.output, record)
                    ok += 1
                    break
                except Exception as exc:
                    last_error = f"attempt_{attempt}: {type(exc).__name__}: {exc}"
                    time.sleep(min(10, 2**attempt))
            else:
                record = steps_to_chain_dict(
                    slide_id,
                    [],
                    report,
                    "test",
                    extraction_status="failed",
                    error=last_error,
                )
                append_record(args.output, record)
                failed += 1
        except Exception as exc:
            append_record(
                args.output,
                {
                    "slide_id": slide_id,
                    "split": "test",
                    "chain-of-thought": [],
                    "node_path": [],
                    "report": "",
                    "extraction_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            failed += 1

    print(f"Done: ok={ok} failed={failed} -> {args.output}")


if __name__ == "__main__":
    main()
