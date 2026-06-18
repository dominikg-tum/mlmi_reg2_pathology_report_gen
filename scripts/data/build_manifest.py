"""Build cases.csv with train/test split from labels xlsx."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_VAL_SIZE = 150
TEST_SIZE = 70
SEED = 42


def load_paths() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def write_example_manifest(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "case_id": "example_001",
            "slide_ids": "be61bc63-d708-4b81-9ea0-2370524e73a8.svs",
            "case_class": "A",
            "split": "train",
            "n_slides": "1",
        },
        {
            "case_id": "example_002",
            "slide_ids": "test_slide.svs",
            "case_class": "B",
            "split": "test",
            "n_slides": "1",
        },
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote example manifest to {out}")


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
    for c in ("slide_ids", "slide_id", "Slide_ID"):
        if c in df.columns:
            slide_col = c
            break
    class_col = "case_class" if "case_class" in df.columns else None

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "slide_ids", "case_class", "split", "n_slides"])
        for i, row in df.iterrows():
            slides = str(row[slide_col]) if slide_col else f"case_{i}"
            cc = str(row[class_col]) if class_col else ""
            split = "test" if i in test_idx else "train"
            n_slides = len(slides.split(",")) if "," in slides else 1
            w.writerow([f"case_{i}", slides, cc, split, n_slides])
    print(f"Wrote {n} cases to {out} (test={test_n})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/manifests/cases.csv")
    parser.add_argument("--example-only", action="store_true")
    args = parser.parse_args()

    if args.example_only:
        write_example_manifest(args.output)
        return

    cfg = load_paths()
    xlsx = Path(cfg["cluster"]["labels_xlsx"])
    if not xlsx.exists():
        print(f"xlsx not found at {xlsx}; writing example manifest")
        write_example_manifest(args.output)
        return
    build_from_xlsx(xlsx, args.output)


if __name__ == "__main__":
    main()
