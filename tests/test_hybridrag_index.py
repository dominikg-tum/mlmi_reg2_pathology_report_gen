import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from extraction.labels_io import load_case_splits, _disk_and_slide_to_case
from memory.hybridrag import (
    HybridRAGMemory,
    compute_corpus_fingerprint,
    get_reference_dir_default,
    load_reference_documents,
    load_report_documents,
    read_corpus_fingerprint,
    read_hybridrag_manifest,
    write_corpus_fingerprint,
    write_hybridrag_manifest,
)


def _labels_dataframe(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "slide_ids": f"disk_{i:04d}.svs",
                "english_reports": f"Report text for slide {i}",
            }
        )
    return pd.DataFrame(rows)


def _write_mini_xlsx(path: Path, n: int = 4) -> None:
    path.write_bytes(b"placeholder")


def _write_split_fixtures(tmp_path: Path, n: int = 6, *, n_test: int = 2) -> tuple[Path, Path]:
    """Write cases.csv + wsi_name_map.csv for disk_0000.. matching _labels_dataframe."""
    cases = tmp_path / "cases.csv"
    name_map = tmp_path / "wsi_name_map.csv"
    with cases.open("w", encoding="utf-8") as handle:
        handle.write("case_id,slide_ids,case_class,disease_label,split,n_slides\n")
        for i in range(n):
            split = "test" if i < n_test else "train"
            handle.write(
                f"p{i:04d},TUM_Uterus_{i:04d}.svs,,label,{split},1\n"
            )
    with name_map.open("w", encoding="utf-8") as handle:
        handle.write(
            "wsi_index,tum_num,slide_id,disk_name,specimen_slide_id,case_key,"
            "block_id,img_id,tum_image_id,disease_label,report_duplicate\n"
        )
        for i in range(n):
            handle.write(
                f"{i},{i:04d},TUM_Uterus_{i:04d}.svs,disk_{i:04d}.svs,"
                f"TUM_Uterus_{i:04d}_p{i:04d}_A.svs,p{i:04d},A,,,label,0\n"
            )
    load_case_splits.cache_clear()
    _disk_and_slide_to_case.cache_clear()
    return cases, name_map


def test_load_report_documents_assigns_train_test_split(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "labels.xlsx"
    _write_mini_xlsx(xlsx, n=6)
    monkeypatch.setattr(pd, "read_excel", lambda _path: _labels_dataframe(6))
    cases, name_map = _write_split_fixtures(tmp_path, n=6, n_test=2)

    train_docs = load_report_documents(
        xlsx, split="train", cases_csv=cases, name_map_csv=name_map
    )
    test_docs = load_report_documents(
        xlsx, split="test", cases_csv=cases, name_map_csv=name_map
    )

    assert len(train_docs) == 4
    assert len(test_docs) == 2
    assert all(doc["page_content"].startswith("Report text") for doc in train_docs + test_docs)
    assert all(doc["metadata"].get("case_key") for doc in train_docs + test_docs)


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

    def fake_build_index(
        self,
        train_reports_path,
        *,
        split="train",
        reference_dir=None,
        include_reference=None,
        force_rebuild=False,
    ):
        self.ensemble_retriever = object()

    monkeypatch.setattr(HybridRAGMemory, "build_index", fake_build_index)

    assert mem.ensure_loaded(manifest_path=manifest) is True
    assert mem.ensemble_retriever is not None


def test_ensure_loaded_respects_nocap_include_reference(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "labels.xlsx"
    _write_mini_xlsx(xlsx, n=3)
    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    manifest = tmp_path / "hybridrag_manifest_nocap.json"
    write_hybridrag_manifest(
        manifest,
        source_path=xlsx,
        chroma_storage=chroma,
        split="train",
        document_count=2,
        variant="nocap",
        include_reference=False,
    )

    seen: dict[str, object] = {}

    def fake_build_index(
        self,
        train_reports_path,
        *,
        split="train",
        reference_dir=None,
        include_reference=None,
        force_rebuild=False,
    ):
        seen["include_reference"] = include_reference
        seen["reference_dir"] = reference_dir
        self.ensemble_retriever = object()

    monkeypatch.setattr(HybridRAGMemory, "build_index", fake_build_index)
    mem = HybridRAGMemory(variant="nocap", chroma_storage=chroma, manifest_path=manifest)
    assert mem.ensure_loaded() is True
    assert seen["include_reference"] is False


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
    assert loaded["variant"] == "cap"
    assert loaded["include_reference"] is True


def test_hybridrag_manifest_nocap_omits_reference_dir(tmp_path: Path):
    manifest = tmp_path / "hybridrag_manifest_nocap.json"
    write_hybridrag_manifest(
        manifest,
        source_path=tmp_path / "labels.xlsx",
        chroma_storage=tmp_path / "chroma_nocap",
        split="train",
        document_count=10,
        report_document_count=10,
        reference_document_count=0,
        variant="nocap",
        include_reference=False,
    )
    loaded = read_hybridrag_manifest(manifest)
    assert loaded is not None
    assert loaded["variant"] == "nocap"
    assert loaded["include_reference"] is False
    assert "reference_dir" not in loaded


def test_resolve_hybridrag_paths_variants():
    from memory.hybridrag import resolve_hybridrag_paths

    nocap = resolve_hybridrag_paths("nocap")
    cap = resolve_hybridrag_paths("cap")
    assert nocap["include_reference"] is False
    assert cap["include_reference"] is True
    assert nocap["chroma_storage"] != cap["chroma_storage"]
    assert nocap["manifest_path"] != cap["manifest_path"]


def test_get_semantic_memory_hybridrag_variants(monkeypatch):
    from memory import base as memory_base

    created: list[str] = []

    class FakeHybrid:
        def __init__(self, *, variant=None, **kwargs):
            created.append(variant)
            self.variant = variant

        def ensure_loaded(self, **kwargs):
            return True

    import memory.hybridrag as hybridrag_mod

    monkeypatch.setattr(hybridrag_mod, "HybridRAGMemory", FakeHybrid)
    # get_semantic_memory imports HybridRAGMemory from memory.hybridrag at call time
    mem_nocap = memory_base.get_semantic_memory("hybridrag")
    mem_cap = memory_base.get_semantic_memory("hybridrag_cap")
    assert created == ["nocap", "cap"]
    assert mem_nocap.variant == "nocap"
    assert mem_cap.variant == "cap"


def test_corpus_fingerprint_stable_and_sensitive_to_corpus(tmp_path: Path):
    source = tmp_path / "labels.xlsx"
    source.write_bytes(b"x")
    ref_dir = tmp_path / "reference"
    report_docs = [
        {
            "page_content": "report A",
            "metadata": {"slide_id": "s1", "source_type": "case_report"},
        }
    ]
    reference_docs = [
        {
            "page_content": "ref A",
            "metadata": {"slide_id": "r1", "source_type": "reference"},
        }
    ]

    fp1 = compute_corpus_fingerprint(
        source_path=source,
        split="train",
        report_docs=report_docs,
        reference_docs=reference_docs,
        reference_dir=ref_dir,
    )
    fp2 = compute_corpus_fingerprint(
        source_path=source,
        split="train",
        report_docs=report_docs,
        reference_docs=reference_docs,
        reference_dir=ref_dir,
    )
    assert fp1["digest"] == fp2["digest"]

    fp_split = compute_corpus_fingerprint(
        source_path=source,
        split="test",
        report_docs=report_docs,
        reference_docs=reference_docs,
        reference_dir=ref_dir,
    )
    assert fp_split["digest"] != fp1["digest"]

    fp_ref = compute_corpus_fingerprint(
        source_path=source,
        split="train",
        report_docs=report_docs,
        reference_docs=reference_docs
        + [{"page_content": "ref B", "metadata": {"slide_id": "r2", "source_type": "reference"}}],
        reference_dir=ref_dir,
    )
    assert fp_ref["digest"] != fp1["digest"]

    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    write_corpus_fingerprint(chroma, fp1)
    assert read_corpus_fingerprint(chroma)["digest"] == fp1["digest"]


def _patch_report_docs_with_fixtures(tmp_path: Path, monkeypatch, *, n: int = 4, n_test: int = 1):
    cases, name_map = _write_split_fixtures(tmp_path, n=n, n_test=n_test)

    def _load(path, *, split="train", cases_csv=None, name_map_csv=None):
        return load_report_documents(
            path,
            split=split,
            cases_csv=cases_csv or cases,
            name_map_csv=name_map_csv or name_map,
        )

    monkeypatch.setattr("memory.hybridrag.load_report_documents", _load)
    return cases, name_map


def test_build_index_rejects_mismatched_fingerprint(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "labels.xlsx"
    _write_mini_xlsx(xlsx)
    monkeypatch.setattr(pd, "read_excel", lambda _path: _labels_dataframe(4))
    _patch_report_docs_with_fixtures(tmp_path, monkeypatch, n=4, n_test=1)

    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    write_corpus_fingerprint(
        chroma,
        {
            "digest": "not-the-real-digest",
            "source_path": str(xlsx.resolve()),
            "split": "train",
            "reference_dir": str((tmp_path / "reference").resolve()),
            "report_document_count": 0,
            "reference_document_count": 0,
        },
    )

    mem = HybridRAGMemory(chroma_storage=chroma)
    with pytest.raises(RuntimeError, match="does not match the current corpus"):
        mem.build_index(str(xlsx), split="train", reference_dir=tmp_path / "reference")


def test_build_index_rejects_missing_fingerprint(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "labels.xlsx"
    _write_mini_xlsx(xlsx)
    monkeypatch.setattr(pd, "read_excel", lambda _path: _labels_dataframe(4))
    _patch_report_docs_with_fixtures(tmp_path, monkeypatch, n=4, n_test=1)

    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    mem = HybridRAGMemory(chroma_storage=chroma)
    with pytest.raises(RuntimeError, match="does not match the current corpus"):
        mem.build_index(str(xlsx), split="train", reference_dir=tmp_path / "reference")


def test_build_index_loads_chroma_when_fingerprint_matches(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "labels.xlsx"
    _write_mini_xlsx(xlsx)
    monkeypatch.setattr(pd, "read_excel", lambda _path: _labels_dataframe(4))
    cases, name_map = _patch_report_docs_with_fixtures(tmp_path, monkeypatch, n=4, n_test=1)

    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    ref_dir = tmp_path / "reference"
    report_docs = load_report_documents(
        xlsx, split="train", cases_csv=cases, name_map_csv=name_map
    )
    fingerprint = compute_corpus_fingerprint(
        source_path=xlsx,
        split="train",
        report_docs=report_docs,
        reference_docs=[],
        reference_dir=None,
        include_reference=False,
    )
    write_corpus_fingerprint(chroma, fingerprint)

    fake_chroma_instance = MagicMock()
    fake_chroma_instance.as_retriever.return_value = MagicMock()
    fake_chroma_cls = MagicMock(return_value=fake_chroma_instance)
    fake_chroma_cls.from_documents = MagicMock()

    chroma_mod = types.ModuleType("langchain_chroma")
    chroma_mod.Chroma = fake_chroma_cls
    community_retrievers = types.ModuleType("langchain_community.retrievers")
    community_retrievers.BM25Retriever = MagicMock()
    community_retrievers.BM25Retriever.from_documents = MagicMock(return_value=MagicMock())
    community_mod = types.ModuleType("langchain_community")
    community_mod.retrievers = community_retrievers
    classic_retrievers = types.ModuleType("langchain_classic.retrievers")
    classic_retrievers.EnsembleRetriever = MagicMock(return_value=MagicMock())
    classic_mod = types.ModuleType("langchain_classic")
    classic_mod.retrievers = classic_retrievers
    core_documents = types.ModuleType("langchain_core.documents")

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    core_documents.Document = _Doc
    core_mod = types.ModuleType("langchain_core")
    core_mod.documents = core_documents

    monkeypatch.setitem(sys.modules, "langchain_chroma", chroma_mod)
    monkeypatch.setitem(sys.modules, "langchain_community", community_mod)
    monkeypatch.setitem(sys.modules, "langchain_community.retrievers", community_retrievers)
    monkeypatch.setitem(sys.modules, "langchain_classic", classic_mod)
    monkeypatch.setitem(sys.modules, "langchain_classic.retrievers", classic_retrievers)
    monkeypatch.setitem(sys.modules, "langchain_core", core_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", core_documents)
    monkeypatch.setattr(HybridRAGMemory, "embeddings", property(lambda self: MagicMock()))

    mem = HybridRAGMemory(chroma_storage=chroma, include_reference=False)
    mem.build_index(str(xlsx), split="train", include_reference=False)

    fake_chroma_cls.assert_called_once()
    fake_chroma_cls.from_documents.assert_not_called()
    assert mem.ensemble_retriever is not None
