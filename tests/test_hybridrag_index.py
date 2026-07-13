import json
from pathlib import Path

import pandas as pd
import pytest
from memory.hybridrag import (
    HybridRAGMemory,
    get_reference_dir_default,
    load_reference_documents,
    load_report_documents,
    read_hybridrag_manifest,
    write_hybridrag_manifest,
)


def _labels_dataframe(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "slide_ids": f"TUM_Uterus_{i:04d}.svs",
                "english_reports": f"Report text for slide {i}",
            }
        )
    return pd.DataFrame(rows)


def _write_mini_xlsx(path: Path, n: int = 4) -> None:
    path.write_bytes(b"placeholder")


def test_load_report_documents_assigns_train_test_split(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "labels.xlsx"
    _write_mini_xlsx(xlsx, n=6)
    monkeypatch.setattr(pd, "read_excel", lambda _path: _labels_dataframe(6))

    train_docs = load_report_documents(xlsx, split="train")
    test_docs = load_report_documents(xlsx, split="test")

    assert len(train_docs) + len(test_docs) == 6
    assert all(doc["page_content"].startswith("Report text") for doc in train_docs + test_docs)


def test_hybridrag_manifest_roundtrip(tmp_path: Path):
    manifest = tmp_path / "hybridrag_manifest.json"
    write_hybridrag_manifest(
        manifest,
        source_path=tmp_path / "labels.xlsx",
        chroma_storage=tmp_path / "chroma_db",
        split="train",
        document_count=12,
    )

    loaded = read_hybridrag_manifest(manifest)
    assert loaded is not None
    assert loaded["split"] == "train"
    assert loaded["document_count"] == 12
    assert loaded["source_path"].endswith("labels.xlsx")


def test_ensure_loaded_uses_manifest_without_langchain(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "labels.xlsx"
    _write_mini_xlsx(xlsx, n=3)
    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    manifest = tmp_path / "hybridrag_manifest.json"
    write_hybridrag_manifest(
        manifest,
        source_path=xlsx,
        chroma_storage=chroma,
        split="train",
        document_count=2,
    )

    mem = HybridRAGMemory(chroma_storage=chroma)

    def fake_build_index(self, train_reports_path, *, split="train", force_rebuild=False):
        self.ensemble_retriever = object()

    monkeypatch.setattr(HybridRAGMemory, "build_index", fake_build_index)

    assert mem.ensure_loaded(manifest_path=manifest) is True
    assert mem.ensemble_retriever is not None


def test_ensure_loaded_false_when_index_missing(tmp_path: Path):
    mem = HybridRAGMemory(chroma_storage=tmp_path / "missing_chroma")
    assert mem.ensure_loaded(manifest_path=tmp_path / "missing_manifest.json") is False


def test_load_reference_documents_from_jsonl(tmp_path: Path):
    ref_root = tmp_path / "reference" / "uterus"
    ref_root.mkdir(parents=True)
    chunk = {
        "id": "test_hyperplasia",
        "title": "Test Hyperplasia",
        "source": "unit test",
        "source_type": "reference",
        "topic": "endometrium",
        "graph_nodes": ["endometrial_hyperplasia_grade"],
        "tier": "local_features",
        "text": "Crowded glands with cytologic atypia suggest AH/EIN.",
    }
    (ref_root / "chunks.jsonl").write_text(json.dumps(chunk) + "\n")

    docs = load_reference_documents(ref_root.parent)
    assert len(docs) == 1
    assert docs[0]["metadata"]["source_type"] == "reference"
    assert docs[0]["metadata"]["slide_id"] == "test_hyperplasia"
    assert "AH/EIN" in docs[0]["page_content"]
    assert docs[0]["metadata"]["graph_nodes"] == ["endometrial_hyperplasia_grade"]


def test_load_reference_documents_missing_dir_returns_empty(tmp_path: Path):
    assert load_reference_documents(tmp_path / "nope") == []


def test_seed_uterus_reference_chunks_load():
    ref_dir = get_reference_dir_default()
    if not (ref_dir / "uterus" / "chunks.jsonl").exists():
        pytest.skip("seed reference chunks not present in this checkout")
    docs = load_reference_documents(ref_dir)
    assert len(docs) >= 10
    assert all(doc["metadata"]["source_type"] == "reference" for doc in docs)


def test_hybridrag_manifest_includes_reference_fields(tmp_path: Path):
    manifest = tmp_path / "hybridrag_manifest.json"
    write_hybridrag_manifest(
        manifest,
        source_path=tmp_path / "labels.xlsx",
        chroma_storage=tmp_path / "chroma_db",
        split="train",
        document_count=22,
        report_document_count=10,
        reference_document_count=12,
        reference_dir=tmp_path / "reference",
    )

    loaded = read_hybridrag_manifest(manifest)
    assert loaded is not None
    assert loaded["reference_document_count"] == 12
    assert loaded["report_document_count"] == 10
    assert loaded["reference_dir"].endswith("reference")
