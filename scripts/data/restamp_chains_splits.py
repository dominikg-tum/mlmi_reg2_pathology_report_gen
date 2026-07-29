"""Restamp chains.jsonl split fields from data/manifests/cases.csv.

Canonical authority: wsi_name_map.csv (UUID → case_key) + cases.csv (case_key → split).
Does not regenerate CoT content.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAINS = REPO_ROOT / "data" / "labels" / "chains.jsonl"
DEFAULT_CASES = REPO_ROOT / "data" / "manifests" / "cases.csv"
DEFAULT_NAME_MAP = REPO_ROOT / "data" / "manifests" / "wsi_name_map.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--name-map", type=Path, default=DEFAULT_NAME_MAP)
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="Backup path (default: <chains>.bak when writing)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any row cannot be mapped to cases.csv",
    )
    args = parser.parse_args()

    from extraction.labels_io import lookup_case_split

    if not args.chains.is_file():
        raise SystemExit(f"chains not found: {args.chains}")
    if not args.cases.is_file():
        raise SystemExit(f"cases.csv not found: {args.cases}")
    if not args.name_map.is_file():
        raise SystemExit(f"name map not found: {args.name_map}")

    cases_s = str(args.cases)
    map_s = str(args.name_map)

    lines = args.chains.read_text(encoding="utf-8").splitlines()
    out_rows: list[dict] = []
    flipped = 0
    unchanged = 0
    unmapped = 0
    new_splits: Counter[str] = Counter()
    flip_examples: list[str] = []

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        old_split = str(row.get("split", "") or "")
        slide_id = str(row.get("slide_id", "") or "")
        lookup = lookup_case_split(slide_id, cases_csv=cases_s, name_map_csv=map_s)
        if not lookup.mapped or not lookup.split:
            unmapped += 1
            print(
                f"WARNING: line {line_no}: unmapped slide_id={slide_id!r} "
                f"(keeping split={old_split!r})",
                file=sys.stderr,
            )
            out_rows.append(row)
            new_splits[old_split or "?"] += 1
            continue

        new_split = lookup.split
        if new_split != old_split:
            flipped += 1
            if len(flip_examples) < 12:
                flip_examples.append(
                    f"  {lookup.case_key}: {old_split!r} -> {new_split!r}"
                )
            row = dict(row)
            row["split"] = new_split
        else:
            unchanged += 1
        out_rows.append(row)
        new_splits[new_split] += 1

    print(f"rows={len(out_rows)} flipped={flipped} unchanged={unchanged} unmapped={unmapped}")
    print(f"split counts after: {dict(new_splits)}")
    if flip_examples:
        print("flip examples:")
        print("\n".join(flip_examples))

    if args.strict and unmapped:
        raise SystemExit(f"--strict: {unmapped} unmapped rows")

    if args.dry_run:
        print("dry-run: no write")
        return

    backup = args.backup if args.backup is not None else Path(str(args.chains) + ".bak")
    shutil.copy2(args.chains, backup)
    print(f"backup -> {backup}")

    tmp = args.chains.with_suffix(args.chains.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.chains)
    print(f"wrote {args.chains}")


if __name__ == "__main__":
    main()
