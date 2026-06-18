import json
from pathlib import Path

import pandas as pd
import pytest
from memory.hybridrag import (
    HybridRAGMemory,
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
