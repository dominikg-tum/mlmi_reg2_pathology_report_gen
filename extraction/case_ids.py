"""Case-level slide_id helpers for SS-LLM multi-WSI inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def parse_slide_ids(raw: str) -> list[str]:
    """Split a comma-separated slide_ids cell into physical WSI ids."""
    if not raw or not str(raw).strip():
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


@dataclass(frozen=True)
class CaseSpec:
    """One GT chains.jsonl row (= one pathology case / eval key)."""

    case_key: str
    physical_slides: list[str]
    split: str = ""


def case_run_dir(runs_dir: Path, case_key: str) -> Path:
    return Path(runs_dir) / case_key


def physical_run_dir(runs_dir: Path, case_key: str, physical_id: str) -> Path:
    return case_run_dir(runs_dir, case_key) / "slides" / physical_id


def load_cases_from_chains(
    chains_path: Path,
    *,
    split: str = "",
) -> list[CaseSpec]:
    """Load unique cases from chains.jsonl (case_key = GT slide_id string)."""
    by_key: dict[str, CaseSpec] = {}
    with Path(chains_path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("extraction_status", "ok") != "ok":
                continue
            case_key = str(raw.get("slide_id", "")).strip()
            if not case_key:
                continue
            row_split = str(raw.get("split", "") or "")
            if split and row_split != split:
                continue
            physical = parse_slide_ids(case_key)
            if not physical:
                continue
            by_key[case_key] = CaseSpec(
                case_key=case_key,
                physical_slides=physical,
                split=row_split,
            )
    return sorted(by_key.values(), key=lambda c: c.case_key)


def case_spec_from_key(case_key: str, *, split: str = "") -> CaseSpec:
    physical = parse_slide_ids(case_key)
    if not physical:
        raise ValueError(f"Empty physical slide list for case_key={case_key!r}")
    return CaseSpec(case_key=case_key, physical_slides=physical, split=split)
