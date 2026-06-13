"""Build HippoRAG 2 fallback index from train-split CoT JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from memory.hipporag2 import HippoRAG2Memory

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic memory index (embedding fallback).")
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "chains.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "memory" / "hipporag_index.json",
    )
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    mem = HippoRAG2Memory(index_path=args.output)
    mem.build_index(str(args.train_jsonl), split=args.split)
    print(f"Indexed {len(mem._steps)} steps -> {args.output}")


if __name__ == "__main__":
    main()
