"""Load slides from xlsx, assign splits, read/write chains.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SLIDE_ID_COLUMN = "slide_ids"
REPORT_COLUMN = "english_reports"
SEED = 42
TEST_SIZE = 70


@dataclass
class SlideRow:
    slide_id: str
    report: str
    split: str
    row_index: int


def assign_splits(n: int, *, seed: int = SEED, test_n: int = TEST_SIZE) -> dict[int, str]:
    """Map row index -> train|test (same logic as scripts/data/build_manifest.py)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    test_count = min(test_n, max(1, n // 3)) if n > 0 else 0
    test_idx = set(indices[:test_count].tolist())
    return {i: ("test" if i in test_idx else "train") for i in range(n)}


def load_slides_from_xlsx(
    xlsx_path: Path,
    *,
    slide_filter: str = "",
    limit: int = 0,
) -> list[SlideRow]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Labels xlsx not found: {xlsx_path}")

    import pandas as pd

    df = pd.read_excel(xlsx_path)
    if SLIDE_ID_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {SLIDE_ID_COLUMN}")
    if REPORT_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {REPORT_COLUMN}")

    splits = assign_splits(len(df))
    rows: list[SlideRow] = []

    for i, row in df.iterrows():
        slide_id = str(row[SLIDE_ID_COLUMN]).strip()
        if not slide_id or slide_id.lower() in ("nan", "n.a.", "na"):
            continue
        if slide_filter and slide_id != slide_filter:
            continue

        report_val = row[REPORT_COLUMN]
        report = "" if pd.isna(report_val) else str(report_val).strip()
        rows.append(
            SlideRow(
                slide_id=slide_id,
                report=report,
                split=splits.get(int(i), "train"),
                row_index=int(i),
            )
        )

    if slide_filter and not rows:
        raise ValueError(f"Slide not found in xlsx: {slide_filter}")

    if limit > 0:
        rows = rows[:limit]
    return rows


def load_existing_slide_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            sid = raw.get("slide_id", "")
            if sid and raw.get("extraction_status", "ok") == "ok":
                ids.add(sid)
    return ids


def write_chains_jsonl(records: list[dict], path: Path, *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
