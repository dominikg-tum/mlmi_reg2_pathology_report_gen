"""Build HybridRAG Chroma + BM25 index from train-split pathology reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from memory.hybridrag import (
    DEFAULT_MANIFEST_PATH,
    HybridRAGMemory,
    get_chroma_storage_default,
    get_labels_xlsx_default,
    load_report_documents,
    write_hybridrag_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths_config() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_paths_config()
    parser = argparse.ArgumentParser(
        description="Build HybridRAG semantic index (Chroma + BM25, train split only)."
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path(cfg["cluster"]["labels_xlsx"]),
        help="Labels spreadsheet with english_reports column.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--chroma-storage",
        type=Path,
        default=None,
        help="Override Chroma persist directory (default: configs/paths.yaml rag.chroma_db_storage).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Write load manifest for inference auto-load.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing Chroma storage and rebuild embeddings.",
    )
    args = parser.parse_args()

    xlsx_path = args.xlsx.expanduser()
    if not xlsx_path.exists():
        raise SystemExit(f"Labels xlsx not found: {xlsx_path}")

    chroma_storage = args.chroma_storage or get_chroma_storage_default()
    documents = load_report_documents(xlsx_path, split=args.split)

    mem = HybridRAGMemory(chroma_storage=chroma_storage)
    mem.build_index(str(xlsx_path), split=args.split, force_rebuild=args.force_rebuild)

    write_hybridrag_manifest(
        args.manifest,
        source_path=xlsx_path,
        chroma_storage=chroma_storage,
        split=args.split,
        document_count=len(documents),
    )
    print(
        f"Indexed {len(documents)} {args.split!r} reports -> {chroma_storage}\n"
        f"Manifest: {args.manifest}"
    )


if __name__ == "__main__":
    main()
