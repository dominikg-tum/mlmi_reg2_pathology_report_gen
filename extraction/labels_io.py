"""Load slides from xlsx, resolve splits from cases.csv, read/write chains.jsonl."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SLIDE_ID_COLUMN = "slide_ids"
REPORT_COLUMN = "english_reports"
SEED = 42
TEST_SIZE = 70

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_CSV = REPO_ROOT / "data" / "manifests" / "cases.csv"
DEFAULT_NAME_MAP_CSV = REPO_ROOT / "data" / "manifests" / "wsi_name_map.csv"


@dataclass
class SlideRow:
    slide_id: str
    report: str
    split: str
    row_index: int


@dataclass(frozen=True)
class CaseSplitLookup:
    """Result of joining a chains/xlsx slide_id string to cases.csv."""

    case_key: str | None
    split: str | None
    mapped: bool


def assign_splits(n: int, *, seed: int = SEED, test_n: int = TEST_SIZE) -> dict[int, str]:
    """Legacy row-index RNG split (tests / historical only — not production).

    Production split authority is ``data/manifests/cases.csv`` via
    :func:`lookup_case_split`.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    test_count = min(test_n, max(1, n // 3)) if n > 0 else 0
    test_idx = set(indices[:test_count].tolist())
    return {i: ("test" if i in test_idx else "train") for i in range(n)}


@lru_cache(maxsize=4)
def load_case_splits(cases_csv: str | None = None) -> dict[str, str]:
    """Map case_id -> train|test from cases.csv."""
    path = Path(cases_csv) if cases_csv else DEFAULT_CASES_CSV
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = (row.get("case_id") or "").strip()
            split = (row.get("split") or "").strip()
            if case_id and split:
                out[case_id] = split
    return out


@lru_cache(maxsize=4)
def _disk_and_slide_to_case(name_map_csv: str | None = None) -> dict[str, str]:
    """Lowercase basename (disk_name or slide_id) -> case_key from wsi_name_map.csv."""
    path = Path(name_map_csv) if name_map_csv else DEFAULT_NAME_MAP_CSV
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            case_key = (row.get("case_key") or "").strip()
            if not case_key:
                continue
            for col in ("disk_name", "slide_id", "specimen_slide_id"):
                val = (row.get(col) or "").strip()
                if val:
                    out[Path(val).name.lower()] = case_key
    return out


def parse_slide_id_tokens(raw: str) -> list[str]:
    """Split a comma/semicolon-separated slide_ids cell into basenames."""
    if not raw or not str(raw).strip():
        return []
    text = str(raw).strip()
    if text.lower() in ("nan", "n.a.", "na", "none"):
        return []
    tokens: list[str] = []
    for part in text.replace(";", ",").split(","):
        tok = part.strip()
        if tok and tok.lower() not in ("nan", "n.a.", "na"):
            tokens.append(Path(tok).name)
    return tokens


def lookup_case_split(
    slide_id_raw: str,
    *,
    cases_csv: str | Path | None = None,
    name_map_csv: str | Path | None = None,
) -> CaseSplitLookup:
    """Map a chains/xlsx ``slide_id`` string to ``case_key`` + ``cases.csv`` split.

    Requires all physical tokens to resolve to the same ``case_key``.
    Uses CSV only (no ``vision`` import) so cluster head / restamp stays light.
    """
    cases_path = str(cases_csv) if cases_csv else None
    map_path = str(name_map_csv) if name_map_csv else None
    tokens = parse_slide_id_tokens(slide_id_raw)
    if not tokens:
        return CaseSplitLookup(case_key=None, split=None, mapped=False)

    name_to_case = _disk_and_slide_to_case(map_path)
    case_keys: list[str] = []
    for tok in tokens:
        case_key = name_to_case.get(tok.lower())
        if not case_key:
            return CaseSplitLookup(case_key=None, split=None, mapped=False)
        case_keys.append(case_key)

    uniq = list(dict.fromkeys(case_keys))
    if len(uniq) != 1:
        return CaseSplitLookup(case_key=None, split=None, mapped=False)

    case_key = uniq[0]
    split = load_case_splits(cases_path).get(case_key)
    if not split:
        return CaseSplitLookup(case_key=case_key, split=None, mapped=False)
    return CaseSplitLookup(case_key=case_key, split=split, mapped=True)


def load_slides_from_xlsx(
    xlsx_path: Path,
    *,
    slide_filter: str = "",
    limit: int = 0,
    cases_csv: Path | None = None,
    name_map_csv: Path | None = None,
) -> list[SlideRow]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Labels xlsx not found: {xlsx_path}")

    import pandas as pd

    df = pd.read_excel(xlsx_path)
    if SLIDE_ID_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {SLIDE_ID_COLUMN}")
    if REPORT_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {REPORT_COLUMN}")

    cases_s = str(cases_csv) if cases_csv else None
    map_s = str(name_map_csv) if name_map_csv else None
    rows: list[SlideRow] = []

    for i, row in df.iterrows():
        slide_id = str(row[SLIDE_ID_COLUMN]).strip()
        if not slide_id or slide_id.lower() in ("nan", "n.a.", "na"):
            continue
        if slide_filter and slide_id != slide_filter:
            continue

        report_val = row[REPORT_COLUMN]
        report = "" if pd.isna(report_val) else str(report_val).strip()
        lookup = lookup_case_split(slide_id, cases_csv=cases_s, name_map_csv=map_s)
        if not lookup.mapped or not lookup.split:
            print(
                f"WARNING: skip xlsx row {i}: no cases.csv split for slide_id={slide_id!r}",
                file=sys.stderr,
            )
            continue
        rows.append(
            SlideRow(
                slide_id=slide_id,
                report=report,
                split=lookup.split,
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
