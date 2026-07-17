"""Build cases.csv with train/test split from wsi_name_map.csv (or labels xlsx fallback)."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_VAL_SIZE = 150
TEST_SIZE = 70
SEED = 42
DEFAULT_NAME_MAP = REPO_ROOT / "data" / "manifests" / "wsi_name_map.csv"


def load_paths() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def write_example_manifest(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "case_id": "p0001",
            "slide_ids": "TUM_Uterus_0001.svs",
            "case_class": "",
            "disease_label": "inflammatory_or_reactive",
            "split": "train",
            "n_slides": "1",
        },
        {
            "case_id": "p0003",
            "slide_ids": "TUM_Uterus_0003.svs,TUM_Uterus_0004.svs",
            "case_class": "",
            "disease_label": "malignant_tumor",
            "split": "test",
            "n_slides": "2",
        },
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote example manifest to {out}")


def build_from_name_map(name_map: Path, out: Path) -> None:
    with name_map.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty name map: {name_map}")

    cases: OrderedDict[str, dict] = OrderedDict()
    for row in rows:
        case_key = (row.get("case_key") or "").strip() or f"slide_{row['slide_id']}"
        slide_id = row["slide_id"].strip()
        report_dup = str(row.get("report_duplicate", "0")).strip() in {"1", "true", "True"}
        disease = (row.get("disease_label") or "").strip()
        if case_key not in cases:
            cases[case_key] = {
                "case_id": case_key,
                "slide_ids": [],
                "disease_label": disease if not report_dup else "",
            }
        cases[case_key]["slide_ids"].append(slide_id)
        if not report_dup and not cases[case_key]["disease_label"]:
            cases[case_key]["disease_label"] = disease
        elif not report_dup and disease:
            cases[case_key]["disease_label"] = disease

    case_list = list(cases.values())
    n = len(case_list)
    rng = __import__("numpy").random.default_rng(SEED)
    indices = rng.permutation(n)
    test_n = min(TEST_SIZE, max(1, n // 3))
    test_idx = set(indices[:test_n].tolist())

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["case_id", "slide_ids", "case_class", "disease_label", "split", "n_slides"]
        )
        for i, case in enumerate(case_list):
            slides = ",".join(case["slide_ids"])
            split = "test" if i in test_idx else "train"
            w.writerow(
                [
                    case["case_id"],
                    slides,
                    "",
                    case["disease_label"],
                    split,
                    len(case["slide_ids"]),
                ]
            )
    print(f"Wrote {n} cases ({sum(len(c['slide_ids']) for c in case_list)} slides) to {out} (test={test_n})")


def build_from_xlsx(xlsx: Path, out: Path) -> None:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("pandas required for xlsx import") from e

    df = pd.read_excel(xlsx)
    n = len(df)
    if n == 0:
        raise SystemExit("Empty xlsx")

    rng = __import__("numpy").random.default_rng(SEED)
    indices = rng.permutation(n)
    test_n = min(TEST_SIZE, max(1, n // 3))
    test_idx = set(indices[:test_n].tolist())

    slide_col = None
    for c in ("slide_ids", "slide_id", "Slide_ID", "anoy_img_id_new"):
        if c in df.columns:
            slide_col = c
            break
    class_col = "case_class" if "case_class" in df.columns else None
    disease_col = "disease label" if "disease label" in df.columns else None

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["case_id", "slide_ids", "case_class", "disease_label", "split", "n_slides"]
        )
        for i, row in df.iterrows():
            slides = str(row[slide_col]) if slide_col else f"case_{i}"
            cc = str(row[class_col]) if class_col else ""
            disease = str(row[disease_col]) if disease_col else ""
            split = "test" if i in test_idx else "train"
            n_slides = len(slides.split(",")) if "," in slides else 1
            w.writerow([f"case_{i}", slides, cc, disease, split, n_slides])
    print(f"Wrote {n} cases to {out} (test={test_n})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/manifests/cases.csv")
    parser.add_argument("--name-map", type=Path, default=DEFAULT_NAME_MAP)
    parser.add_argument("--example-only", action="store_true")
    args = parser.parse_args()

    if args.example_only:
        write_example_manifest(args.output)
        return

    if args.name_map.is_file():
        build_from_name_map(args.name_map, args.output)
        return

    cfg = load_paths()
    xlsx = Path(cfg["cluster"]["labels_xlsx"])
    if not xlsx.exists():
        print(f"name map and xlsx missing; writing example manifest")
        write_example_manifest(args.output)
        return
    build_from_xlsx(xlsx, args.output)


if __name__ == "__main__":
    main()
