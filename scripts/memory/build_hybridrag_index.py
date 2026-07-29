"""Build HybridRAG Chroma + BM25 index from train reports (+ optional CAP refs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from memory.hybridrag import (
    HybridRAGMemory,
    load_reference_documents,
    load_report_documents,
    normalize_hybridrag_variant,
    resolve_hybridrag_paths,
    write_hybridrag_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths_config() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_paths_config()
    parser = argparse.ArgumentParser(
        description=(
            "Build HybridRAG semantic index (Chroma + BM25). "
            "Use --variant nocap|cap for dual ablation stores."
        )
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path(cfg["cluster"]["labels_xlsx"]),
        help="Labels spreadsheet with english_reports column.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--variant",
        choices=["nocap", "cap"],
        default="nocap",
        help=(
            "nocap = train reports only (baseline b2); "
            "cap = reports + CAP/WHO reference chunks (baseline b2_cap). "
            "Sets chroma + manifest from configs/paths.yaml."
        ),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Override reference chunk root (cap variant only).",
    )
    parser.add_argument(
        "--chroma-storage",
        type=Path,
        default=None,
        help="Override Chroma persist directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override load manifest path.",
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

    variant = normalize_hybridrag_variant(args.variant)
    paths = resolve_hybridrag_paths(variant)
    include_reference = bool(paths["include_reference"])
    reference_dir = args.reference_dir if include_reference else None
    if include_reference and reference_dir is None:
        reference_dir = paths["reference_dir"]

    chroma_storage = args.chroma_storage or paths["chroma_storage"]
    manifest_path = args.manifest or paths["manifest_path"]

    report_docs = load_report_documents(xlsx_path, split=args.split)
    reference_docs = (
        load_reference_documents(reference_dir) if include_reference else []
    )
    if include_reference and not reference_docs:
        print(
            f"WARNING: variant=cap but no reference chunks under {reference_dir}",
            flush=True,
        )
    total = len(report_docs) + len(reference_docs)

    mem = HybridRAGMemory(
        chroma_storage=chroma_storage,
        variant=variant,
        manifest_path=manifest_path,
        include_reference=include_reference,
    )
    mem.build_index(
        str(xlsx_path),
        split=args.split,
        reference_dir=reference_dir,
        include_reference=include_reference,
        force_rebuild=args.force_rebuild,
    )

    write_hybridrag_manifest(
        manifest_path,
        source_path=xlsx_path,
        chroma_storage=Path(chroma_storage),
        split=args.split,
        document_count=total,
        report_document_count=len(report_docs),
        reference_document_count=len(reference_docs),
        reference_dir=Path(reference_dir) if reference_dir is not None else None,
        variant=variant,
        include_reference=include_reference,
    )
    print(
        f"variant={variant} include_reference={include_reference}\n"
        f"Indexed {len(report_docs)} {args.split!r} reports + "
        f"{len(reference_docs)} reference chunks -> {chroma_storage}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
