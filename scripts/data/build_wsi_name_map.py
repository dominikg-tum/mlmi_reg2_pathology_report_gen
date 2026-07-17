"""Build data/manifests/wsi_name_map.csv from TUM_Uterus_name_mapping.xlsx.

Canonical slide_id is anoy_img_id_new (TUM_Uterus_XXXX.svs). disk_name is the
UUID .svs filename under cluster.data_dir. Offline jobs index rows 0..N-1 in
this CSV order (not filesystem sort).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XLSX = REPO_ROOT / "data" / "TUM_Uterus_name_mapping.xlsx"
DEFAULT_OUT = REPO_ROOT / "data" / "manifests" / "wsi_name_map.csv"

_SPECIMEN_RE = re.compile(
    r"(?P<slide>TUM_Uterus_\d+)_p(?P<case>\d+)_(?P<block>.+)\.svs$",
    re.IGNORECASE,
)


def build_rows(xlsx: Path) -> list[dict[str, str]]:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("pandas + openpyxl required: uv pip install pandas openpyxl") from e

    df = pd.read_excel(xlsx, sheet_name="seperatable_reports")
    required = ("anoy_img_id_old", "anoy_img_id_new", "anoy_img_id_with_Specimen_Block_slide_ID")
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col!r} in {xlsx}")

    rows: list[dict[str, str]] = []
    for i, row in df.iterrows():
        disk_name = str(row["anoy_img_id_old"]).strip()
        slide_id = str(row["anoy_img_id_new"]).strip()
        specimen = str(row["anoy_img_id_with_Specimen_Block_slide_ID"]).strip()
        skip_raw = row.get("Unnamed: 12", "")
        report_dup = "SKIPPED" in str(skip_raw)

        case_key = ""
        block_id = ""
        match = _SPECIMEN_RE.match(specimen)
        if match:
            case_key = f"p{match.group('case')}"
            block_id = match.group("block")

        tum_num = ""
        m_num = re.search(r"TUM_Uterus_(\d+)", slide_id, re.IGNORECASE)
        if m_num:
            tum_num = m_num.group(1)

        rows.append(
            {
                "wsi_index": str(len(rows)),
                "tum_num": tum_num.zfill(4) if tum_num else "",
                "slide_id": slide_id,
                "disk_name": disk_name,
                "specimen_slide_id": specimen,
                "case_key": case_key,
                "block_id": block_id,
                "img_id": str(row.get("img_ID", "")).strip(),
                "tum_image_id": str(row.get("ImageId in TUM", "")).strip(),
                "disease_label": str(row.get("disease label", "")).strip(),
                "report_duplicate": "1" if report_dup else "0",
            }
        )

    rows.sort(key=lambda r: (int(r["tum_num"]) if r["tum_num"].isdigit() else 10**9, r["slide_id"]))
    for i, row in enumerate(rows):
        row["wsi_index"] = str(i)
    return rows


def write_csv(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wsi_index",
        "tum_num",
        "slide_id",
        "disk_name",
        "specimen_slide_id",
        "case_key",
        "block_id",
        "img_id",
        "tum_image_id",
        "disease_label",
        "report_duplicate",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    n_dup = sum(1 for r in rows if r["report_duplicate"] == "1")
    n_cases = len({r["case_key"] for r in rows if r["case_key"]})
    print(f"Wrote {len(rows)} slides ({n_cases} cases, {n_dup} report_duplicate) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.xlsx.is_file():
        raise SystemExit(f"xlsx not found: {args.xlsx}")
    write_csv(build_rows(args.xlsx), args.output)


if __name__ == "__main__":
    main()
