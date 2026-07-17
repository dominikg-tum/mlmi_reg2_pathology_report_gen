"""UUID <-> TUM_Uterus_XXXX.svs name map for offline WSI indexing.

Source of truth on disk: data/manifests/wsi_name_map.csv
(regenerated from data/TUM_Uterus_name_mapping.xlsx via scripts/data/build_wsi_name_map.py).
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME_MAP_CSV = REPO_ROOT / "data" / "manifests" / "wsi_name_map.csv"


@dataclass(frozen=True)
class WsiNameRow:
    wsi_index: int
    tum_num: str
    slide_id: str
    disk_name: str
    specimen_slide_id: str
    case_key: str
    block_id: str
    img_id: str
    tum_image_id: str
    disease_label: str
    report_duplicate: bool


def _parse_row(raw: dict[str, str]) -> WsiNameRow:
    return WsiNameRow(
        wsi_index=int(raw["wsi_index"]),
        tum_num=str(raw.get("tum_num", "")).strip(),
        slide_id=str(raw["slide_id"]).strip(),
        disk_name=str(raw["disk_name"]).strip(),
        specimen_slide_id=str(raw.get("specimen_slide_id", "")).strip(),
        case_key=str(raw.get("case_key", "")).strip(),
        block_id=str(raw.get("block_id", "")).strip(),
        img_id=str(raw.get("img_id", "")).strip(),
        tum_image_id=str(raw.get("tum_image_id", "")).strip(),
        disease_label=str(raw.get("disease_label", "")).strip(),
        report_duplicate=str(raw.get("report_duplicate", "0")).strip() in {"1", "true", "True"},
    )


@lru_cache(maxsize=4)
def load_wsi_name_map(csv_path: str | None = None) -> tuple[WsiNameRow, ...]:
    path = Path(csv_path) if csv_path else DEFAULT_NAME_MAP_CSV
    if not path.is_file():
        return ()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = tuple(_parse_row(raw) for raw in reader)
    return rows


def name_map_available(csv_path: str | None = None) -> bool:
    return bool(load_wsi_name_map(csv_path))


def _lookup_maps(
    rows: tuple[WsiNameRow, ...],
) -> tuple[dict[str, WsiNameRow], dict[str, WsiNameRow]]:
    by_slide: dict[str, WsiNameRow] = {}
    by_disk: dict[str, WsiNameRow] = {}
    for row in rows:
        by_slide[row.slide_id.lower()] = row
        by_disk[row.disk_name.lower()] = row
        if row.specimen_slide_id:
            by_slide[row.specimen_slide_id.lower()] = row
    return by_slide, by_disk


def row_for_name(name: str, *, csv_path: str | None = None) -> WsiNameRow | None:
    """Resolve a basename (UUID or TUM_Uterus_*) to its mapping row."""
    key = Path(name).name.lower()
    rows = load_wsi_name_map(csv_path)
    if not rows:
        return None
    by_slide, by_disk = _lookup_maps(rows)
    return by_slide.get(key) or by_disk.get(key)


def canonical_slide_id(name_or_path: str | Path, *, csv_path: str | None = None) -> str:
    """Return TUM_Uterus_XXXX.svs when mapped; otherwise the input basename."""
    basename = Path(name_or_path).name
    row = row_for_name(basename, csv_path=csv_path)
    return row.slide_id if row is not None else basename


def disk_filename(name_or_path: str | Path, *, csv_path: str | None = None) -> str:
    """Return the on-disk UUID .svs name when mapped; otherwise the input basename."""
    basename = Path(name_or_path).name
    row = row_for_name(basename, csv_path=csv_path)
    return row.disk_name if row is not None else basename


def mapped_slide_ids(*, csv_path: str | None = None) -> list[str]:
    return [row.slide_id for row in load_wsi_name_map(csv_path)]


def _find_named_svs(data_dir: Path, basename: str) -> Path | None:
    """Locate one basename under data_dir without a full-tree Python rglob."""
    if not basename or not data_dir.is_dir():
        return None
    direct = data_dir / basename
    if direct.is_file():
        return direct
    try:
        proc = subprocess.run(
            ["find", str(data_dir), "-name", basename, "-type", "f", "-print", "-quit"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        matches = list(data_dir.rglob(basename))
        return matches[0] if matches else None
    line = (proc.stdout or "").strip().splitlines()
    if line:
        return Path(line[0])
    return None


def locate_svs_path(
    data_dir: Path,
    row: WsiNameRow,
    *,
    index: dict[str, Path] | None = None,
) -> Path:
    """Find the .svs file for a mapping row under data_dir (disk UUID first)."""
    for candidate in (row.disk_name, row.slide_id, row.specimen_slide_id):
        if not candidate:
            continue
        if index is not None:
            hit = index.get(candidate.lower())
            if hit is not None:
                return hit
            continue
        hit = _find_named_svs(data_dir, candidate)
        if hit is not None:
            return hit
    raise FileNotFoundError(
        f"No .svs for slide_id={row.slide_id!r} disk_name={row.disk_name!r} under {data_dir}"
    )


@lru_cache(maxsize=2)
def _svs_basename_index(data_dir_str: str) -> dict[str, Path]:
    data_dir = Path(data_dir_str)
    if not data_dir.is_dir():
        return {}
    try:
        proc = subprocess.run(
            ["find", str(data_dir), "-name", "*.svs", "-type", "f"],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {path.name.lower(): path for path in sorted(data_dir.rglob("*.svs"))}
    out: dict[str, Path] = {}
    for line in (proc.stdout or "").splitlines():
        path = Path(line.strip())
        if path.name:
            out[path.name.lower()] = path
    return out


def resolve_mapped_wsi_files(
    data_dir: Path,
    *,
    slide: str = "",
    limit: int = 0,
    wsi_index: int | None = None,
    csv_path: str | None = None,
) -> list[Path] | None:
    """Resolve via name map. Returns None if the CSV is missing (caller falls back)."""
    rows = load_wsi_name_map(csv_path)
    if not rows:
        return None

    # Single-slide / array-index paths: avoid scanning the whole WSI tree.
    if slide or wsi_index is not None:
        if slide:
            row = row_for_name(slide, csv_path=csv_path)
            if row is None:
                hit = _find_named_svs(data_dir, Path(slide).name)
                if hit is not None:
                    return [hit]
                raise FileNotFoundError(
                    f"slide={slide!r} not in name map and not under {data_dir}"
                )
            return [locate_svs_path(data_dir, row)]
        if wsi_index is not None:
            if wsi_index < 0 or wsi_index >= len(rows):
                raise IndexError(
                    f"wsi_index={wsi_index} out of range for {len(rows)} mapped slides"
                )
            return [locate_svs_path(data_dir, rows[wsi_index])]

    index = _svs_basename_index(str(data_dir.resolve()))
    selected = list(rows)
    if limit > 0:
        selected = selected[:limit]
    return [locate_svs_path(data_dir, row, index=index) for row in selected]
