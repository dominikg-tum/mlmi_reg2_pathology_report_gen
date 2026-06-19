"""Build UUID (xlsx slide_ids) → TUM_Uterus_XXXX.svs mapping manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from data.case_slides import parse_slide_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "manifests" / "wsi_id_map.json"


def load_paths() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def sorted_tum_svs(data_dir: Path) -> list[str]:
    files = sorted(data_dir.glob("TUM_Uterus_*.svs"))
    return [p.name for p in files]


def load_xlsx_slide_rows(xlsx_path: Path) -> list[tuple[str, list[str]]]:
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    if "slide_ids" not in df.columns:
        raise ValueError(f"Missing slide_ids column in {xlsx_path}")

    rows: list[tuple[str, list[str]]] = []
    for _, row in df.iterrows():
        raw = row["slide_ids"]
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        case_key = str(raw).strip()
        if not case_key or case_key.lower() in ("nan", "n.a.", "na"):
            continue
        uuids = parse_slide_ids(case_key)
        if uuids:
            rows.append((case_key, uuids))
    return rows


def build_sequential_map(
    xlsx_rows: list[tuple[str, list[str]]],
    tum_names: list[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    idx = 0
    for _case_key, uuids in xlsx_rows:
        for uuid in uuids:
            if uuid in mapping:
                raise ValueError(f"Duplicate UUID in xlsx: {uuid}")
            if idx >= len(tum_names):
                raise ValueError(
                    f"Ran out of TUM slides at UUID {uuid} "
                    f"(need {idx + 1}, have {len(tum_names)})"
                )
            mapping[uuid] = tum_names[idx]
            idx += 1
    return mapping


@dataclass
class ValidationReport:
    xlsx_cases: int
    xlsx_slide_refs: int
    tum_on_disk: int
    mapped_pairs: int
    unmapped_tum: list[str]
    duplicate_uuids: list[str]
    report_parts_cases: int | None
    report_parts_mismatch: int

    def print_summary(self) -> None:
        print(f"xlsx cases:           {self.xlsx_cases}")
        print(f"xlsx slide refs:      {self.xlsx_slide_refs}")
        print(f"TUM .svs on disk:     {self.tum_on_disk}")
        print(f"mapped UUID→TUM:      {self.mapped_pairs}")
        print(f"unmapped TUM files:   {len(self.unmapped_tum)}")
        if self.unmapped_tum[:5]:
            print(f"  first unmapped:     {', '.join(self.unmapped_tum[:5])}")
        if self.duplicate_uuids:
            print(f"duplicate UUIDs:      {self.duplicate_uuids}")
        if self.report_parts_cases is not None:
            print(f"report_parts cases:   {self.report_parts_cases}")
            print(f"n_slides mismatches:  {self.report_parts_mismatch}")


def validate_mapping(
    xlsx_rows: list[tuple[str, list[str]]],
    tum_names: list[str],
    mapping: dict[str, str],
    report_parts_path: Path | None,
) -> ValidationReport:
    mapped_tum = set(mapping.values())
    unmapped = [n for n in tum_names if n not in mapped_tum]

    seen: set[str] = set()
    duplicates: list[str] = []
    for _case, uuids in xlsx_rows:
        for u in uuids:
            if u in seen:
                duplicates.append(u)
            seen.add(u)

    rp_cases = None
    rp_mismatch = 0
    if report_parts_path and report_parts_path.exists():
        records = json.loads(report_parts_path.read_text())
        if isinstance(records, list):
            rp_cases = len(records)
            xlsx_by_first_uuid: dict[str, int] = {}
            for _case, uuids in xlsx_rows:
                xlsx_by_first_uuid[uuids[0]] = len(uuids)
            for rec in records:
                sid = rec.get("slide_id", "")
                first = parse_slide_ids(sid)[0] if sid else ""
                if first in xlsx_by_first_uuid:
                    expected = xlsx_by_first_uuid[first]
                    actual = len(parse_slide_ids(sid))
                    if expected != actual:
                        rp_mismatch += 1

    total_refs = sum(len(u) for _, u in xlsx_rows)
    return ValidationReport(
        xlsx_cases=len(xlsx_rows),
        xlsx_slide_refs=total_refs,
        tum_on_disk=len(tum_names),
        mapped_pairs=len(mapping),
        unmapped_tum=unmapped,
        duplicate_uuids=duplicates,
        report_parts_cases=rp_cases,
        report_parts_mismatch=rp_mismatch,
    )


def write_manifest(
    mapping: dict[str, str],
    output: Path,
    *,
    strategy: str,
    validation: ValidationReport,
) -> None:
    payload = {
        "strategy": strategy,
        "mapping": mapping,
        "meta": {
            "xlsx_cases": validation.xlsx_cases,
            "xlsx_slide_refs": validation.xlsx_slide_refs,
            "tum_on_disk": validation.tum_on_disk,
            "mapped_pairs": validation.mapped_pairs,
            "unmapped_tum": validation.unmapped_tum,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UUID → TUM_Uterus WSI id map.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--report-parts",
        type=Path,
        default=None,
        help="Optional report_parts_extracted.json for cross-check",
    )
    parser.add_argument("--validate", action="store_true", help="Print validation report only")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing output")
    args = parser.parse_args()

    cfg = load_paths()
    xlsx_path = args.xlsx or Path(cfg["cluster"]["labels_xlsx"])
    data_dir = args.data_dir or Path(cfg["cluster"]["data_dir"])
    report_parts = args.report_parts
    if report_parts is None:
        rp_cfg = cfg.get("extraction", {}).get("report_parts_json", "")
        if rp_cfg:
            report_parts = Path(rp_cfg)
        else:
            report_parts = Path("/mnt/projects/mlmi/reg2/report_parts_extracted.json")

    if not xlsx_path.exists():
        raise SystemExit(f"Missing xlsx: {xlsx_path}")
    if not data_dir.is_dir():
        raise SystemExit(f"Missing data dir: {data_dir}")

    xlsx_rows = load_xlsx_slide_rows(xlsx_path)
    tum_names = sorted_tum_svs(data_dir)
    mapping = build_sequential_map(xlsx_rows, tum_names)
    validation = validate_mapping(xlsx_rows, tum_names, mapping, report_parts)

    print("=== WSI ID map validation ===")
    validation.print_summary()
    print(f"strategy: sequential xlsx-order → sorted TUM_Uterus_*.svs")

    if validation.duplicate_uuids:
        raise SystemExit("Duplicate UUIDs in xlsx — cannot build map")

    if args.validate and args.dry_run:
        return

    if args.dry_run:
        print(f"(dry-run: would write {len(mapping)} entries -> {args.output})")
        return

    write_manifest(mapping, args.output, strategy="sequential_xlsx_order", validation=validation)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
