"""Build HybridRAG Chroma + BM25 index from train-split chains.jsonl reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from memory.hybridrag import HybridRAGMemory, get_chroma_storage_default

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HybridRAG index from chains.jsonl report field (train split only)."
    )
    parser.add_argument(
        "--chains",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "chains.jsonl",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chroma persist directory (default: configs/paths.yaml rag.chroma_db_storage)",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    chroma_path = args.output or get_chroma_storage_default()
    mem = HybridRAGMemory(chroma_storage=chroma_path)
    mem.build_index_from_chains(args.chains, split=args.split, force_rebuild=args.force_rebuild)
    n_docs = len(mem._documents_from_chroma())
    print(f"Indexed {n_docs} train reports -> {chroma_path}")


if __name__ == "__main__":
    main()
