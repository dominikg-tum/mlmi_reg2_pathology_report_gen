"""
HybridRAG semantic memory

Baseline RAG system for pipeline tests and ablation study.
Combines semantic vector search (ChromaDB + PubMedBERT) with
lexical keyword search (BM25).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
import yaml

from graph.schema import Node, Tier
import pandas as pd

from extraction.labels_io import assign_splits
from memory.base import SemanticMemory

# Path to repository root directory
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "memory" / "hybridrag_manifest.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "data" / "memory" / "reference"
ReportDocument = dict[str, Any]

_REFERENCE_CHUNK_REQUIRED = frozenset({"id", "title", "text", "source", "source_type"})


class HybridRAGMemory(SemanticMemory):
    """
    HybridRAG memory component

    Combines semantic vector search (ChromaDB + PubMedBERT) with
    lexical keyword search (BM25). Vector search captures general conceptual meaning
    of a pathology report. BM25 ensures that highly specific biomarkers
    and abbreviations (e.g., "P40", "WT1") are not missed due to vector dilution.

    Index contents: train-split case reports plus optional reference chunks
    under ``data/memory/reference/**/*.jsonl``.
    """

    def __init__(self, *, chroma_storage: str | Path | None = None):
        """
        Initializes the HybridRAGMemory, setting up storage paths and the
        domain-specific embedding model.
        """
        self.chroma_storage = (
            Path(chroma_storage) if chroma_storage is not None else get_chroma_storage_default()
        )
        self.vector_storage = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.vector_retriever = None
        self._embeddings = None

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            # NeuML/pubmedbert-base-embeddings or BAAI/bge-m3 for pathology
            self._embeddings = HuggingFaceEmbeddings(
                model_name="NeuML/pubmedbert-base-embeddings"
            )
        return self._embeddings

    def build_index(
        self,
        train_reports_path: str,
        *,
        split: str = "train",
        reference_dir: str | Path | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """
        Builds or loads the ensemble index containing both Vector and BM25 databases.

        Reads pathology reports (train split) and optional reference JSONL chunks,
        converts them into Document objects, and persists the Chroma vector database.

        Args:
            train_reports_path: Path to the Excel file containing case reports.
            split: Which dataset split to index from the spreadsheet (default: train).
            reference_dir: Root directory scanned for ``**/*.jsonl`` reference chunks.
            force_rebuild: Delete existing Chroma storage and rebuild embeddings.

        Raises:
            FileNotFoundError: If the specified Excel file does not exist.
            RuntimeError: If an existing Chroma index does not match the current
                corpus fingerprint and ``force_rebuild`` is False.
        """
        ref_dir = Path(reference_dir) if reference_dir is not None else get_reference_dir_default()
        source_path = Path(train_reports_path)
        report_docs = load_report_documents(source_path, split=split)
        reference_docs = load_reference_documents(ref_dir)
        corpus_fingerprint = compute_corpus_fingerprint(
            source_path=source_path,
            split=split,
            report_docs=report_docs,
            reference_docs=reference_docs,
            reference_dir=ref_dir,
        )

        # Delete existing index if force_build
        if force_rebuild and os.path.exists(self.chroma_storage):
            if self.vector_storage is not None:
                try:
                    self.vector_storage._client.close()
                except Exception:
                    pass
                self.vector_storage = None
                self.vector_retriever = None
                self.ensemble_retriever = None
                self.bm25_retriever = None
            gc.collect()
            shutil.rmtree(self.chroma_storage)

        # Load existing Chroma only when corpus fingerprint matches; otherwise rebuild
        # (force_rebuild) or require an explicit rebuild.
        chroma_exists = os.path.exists(self.chroma_storage)
        if chroma_exists:
            stored_fingerprint = read_corpus_fingerprint(self.chroma_storage)
            stored_digest = stored_fingerprint.get("digest") if stored_fingerprint else None
            if stored_digest != corpus_fingerprint["digest"]:
                raise RuntimeError(
                    "RAG: Existing Chroma index does not match the current corpus "
                    f"(source={source_path}, split={split!r}, reference_dir={ref_dir}). "
                    "Rebuild with force_rebuild=True or: "
                    "python -m scripts.memory.build_hybridrag_index --force-rebuild"
                )

        documents = _as_langchain_documents(report_docs + reference_docs)

        from langchain_chroma import Chroma
        from langchain_community.retrievers import BM25Retriever
        from langchain_classic.retrievers import EnsembleRetriever

        if chroma_exists:
            self.vector_storage = Chroma(
                persist_directory=str(self.chroma_storage),
                embedding_function=self.embeddings,
            )
        else:
            self.vector_storage = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=str(self.chroma_storage),
            )
            write_corpus_fingerprint(self.chroma_storage, corpus_fingerprint)

        # Create retriever in initialization progress
        self.vector_retriever = self.vector_storage.as_retriever()

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=clean_tokenize)

        # Combining semantic and BM25 retriever into ensemble
        # Could also do ablation study on weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.65, 0.35],
        )

    def ensure_loaded(self, *, manifest_path: Path | None = None) -> bool:
        """Load an existing on-disk Chroma index + BM25 ensemble when available."""
        if self.ensemble_retriever is not None:
            return True

        manifest = read_hybridrag_manifest(manifest_path)
        if manifest:
            chroma_path = Path(manifest["chroma_storage"])
            source_path = Path(manifest["source_path"])
            if chroma_path.exists() and source_path.exists():
                self.chroma_storage = chroma_path
                ref_dir = manifest.get("reference_dir")
                self.build_index(
                    str(source_path),
                    split=str(manifest.get("split", "train")),
                    reference_dir=ref_dir,
                )
                return self.ensemble_retriever is not None

        if self.chroma_storage.exists():
            source_path = get_labels_xlsx_default()
            if source_path.exists():
                self.build_index(str(source_path), split="train")
                return self.ensemble_retriever is not None
        return False

    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str:
        """
        Retrieves most relevant context from the index based on query and node.

        Adjusts number of returned documents (k) based on retrieval depth.
        Enriches the base query with context from the graph.

        Args:
            node: Current node in the diagnostic graph.
            query: Raw search string.
            k: Base number of documents to retrieve.

        Returns:
            A formatted string concatenating the top-k retrieved documents.

        Raises:
            RuntimeError: If called before ``build_index()`` is executed.
        """
        if self.ensemble_retriever is None and not self.ensure_loaded():
            raise RuntimeError(
                "RAG: HybridRAG index not loaded. Build it first with: "
                "python -m scripts.memory.build_hybridrag_index"
            )

        # Coarse zoom → fewer docs; fine zoom → more docs
        effective_k = _effective_k_for_node(node, k)

        # Update query to include node context
        enriched_query = f"[{node.tier.value}] {node.question} {query}".strip()

        # Update retrievers to search for top k matches
        self.bm25_retriever.k = effective_k
        self.vector_retriever.search_kwargs = {"k": effective_k}

        # Invoke query; fetch extra candidates when reranking reference chunks
        fetch_k = effective_k * 2 if _prefer_reference_for_node(node) else effective_k
        self.bm25_retriever.k = fetch_k
        self.vector_retriever.search_kwargs = {"k": fetch_k}
        results = self.ensemble_retriever.invoke(enriched_query)
        results = _rerank_for_node(results, node)[:effective_k]

        formatted_results = []
        for i, doc in enumerate(results):
            source_id = doc.metadata.get("slide_id", "Unknown")
            source_type = doc.metadata.get("source_type", "case_report")
            title = doc.metadata.get("title", "")
            title_suffix = f" | {title}" if title else ""
            formatted_results.append(
                f"[Document {i + 1} | Source: {source_id} | Type: {source_type}"
                f"{title_suffix} | Node: {node.id} | Tier: {node.tier.value}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted_results)


def _effective_k_for_node(node: Node, k: int) -> int:
    """Map graph zoom tier to retrieval breadth."""
    zoom = node.zoom_level.value if node.zoom_level is not None else "20x"
    if zoom == "5x":
        return max(1, k // 2)
    if zoom in ("20x", "40x"):
        return min(k * 2, 20)
    return k


def _prefer_reference_for_node(node: Node) -> bool:
    """Local/integration nodes benefit most from textbook-style reference chunks."""
    return node.tier in (Tier.LOCAL_FEATURES, Tier.INTEGRATION)


def _rerank_for_node(results: list[Any], node: Node) -> list[Any]:
    """Boost reference chunks on diagnostic nodes; keep ensemble order otherwise."""
    if not _prefer_reference_for_node(node):
        return results

    def sort_key(doc: Any) -> tuple[int, int]:
        is_reference = doc.metadata.get("source_type") == "reference"
        node_match = 1 if node.id in (doc.metadata.get("graph_nodes") or []) else 0
        return (1 if is_reference else 0, node_match)

    return sorted(results, key=sort_key, reverse=True)


def clean_tokenize(text: str) -> list[str]:
    """
    Custom tokenizer for BM25 lexical search.
    Strips punctuation and converts text to lowercase.

    Args:
        text: The raw text string to tokenize.

    Returns:
        A list of cleaned, lowercase word tokens.
    """
    return re.findall(r"\w+", text.lower())


def load_config() -> dict[str, Any]:
    """Loads the main path configuration YAML file."""
    path = REPO_ROOT / "configs" / "paths.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def get_chroma_storage_default() -> Path:
    """Retrieves the default path for Chroma database from the config."""
    cfg = load_config()
    return Path(cfg["rag"]["chroma_db_storage"])


def get_reference_dir_default() -> Path:
    """Default reference chunk root (repo-relative path in configs/paths.yaml)."""
    cfg = load_config()
    raw = cfg.get("rag", {}).get("reference_dir", "data/memory/reference")
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_labels_xlsx_default() -> Path:
    """Default pathology reports spreadsheet (cluster labels xlsx)."""
    cfg = load_config()
    return Path(cfg["cluster"]["labels_xlsx"])


def load_report_documents(
    train_reports_path: str | Path,
    *,
    split: str = "train",
) -> list[ReportDocument]:
    """Load train-split report documents from the labels xlsx."""
    path = Path(train_reports_path)
    if not path.exists():
        raise FileNotFoundError(f"RAG: Reports Excel not found at '{path}'")

    df = pd.read_excel(path)
    if "english_reports" not in df.columns:
        raise ValueError("RAG: labels xlsx missing column 'english_reports'")

    splits = assign_splits(len(df))
    if "split" not in df.columns:
        df = df.copy()
        df["split"] = [splits.get(int(i), "train") for i in range(len(df))]

    df = df.dropna(subset=["english_reports"])
    df = df[df["split"] == split]

    documents: list[ReportDocument] = []
    for index, row in df.iterrows():
        text = str(row["english_reports"])
        slide_id = str(row.get("slide_ids", index))
        documents.append(
            {
                "page_content": text,
                "metadata": {
                    "slide_id": slide_id,
                    "source_type": "case_report",
                },
            }
        )
    return documents


def load_reference_documents(reference_dir: str | Path | None = None) -> list[ReportDocument]:
    """Load curated reference chunks from ``**/*.jsonl`` under reference_dir."""
    root = Path(reference_dir) if reference_dir is not None else get_reference_dir_default()
    if not root.exists():
        return []

    documents: list[ReportDocument] = []
    for jsonl_path in sorted(root.rglob("*.jsonl")):
        for line_no, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"RAG: invalid JSON in {jsonl_path}:{line_no}: {exc}"
                ) from exc
            documents.append(_reference_chunk_to_document(chunk, jsonl_path, line_no))
    return documents


def _reference_chunk_to_document(
    chunk: dict[str, Any],
    source_path: Path,
    line_no: int,
) -> ReportDocument:
    missing = _REFERENCE_CHUNK_REQUIRED - chunk.keys()
    if missing:
        raise ValueError(
            f"RAG: reference chunk at {source_path}:{line_no} missing fields: {sorted(missing)}"
        )
    if chunk.get("source_type") != "reference":
        raise ValueError(
            f"RAG: reference chunk at {source_path}:{line_no} must have source_type='reference'"
        )

    chunk_id = str(chunk["id"])
    title = str(chunk["title"])
    body = str(chunk["text"]).strip()
    if not body:
        raise ValueError(f"RAG: reference chunk {chunk_id!r} has empty text")

    header = f"{title}\nSource: {chunk['source']}\n\n"
    graph_nodes = chunk.get("graph_nodes") or []
    if not isinstance(graph_nodes, list):
        raise ValueError(f"RAG: reference chunk {chunk_id!r} graph_nodes must be a list")

    return {
        "page_content": header + body,
        "metadata": {
            "slide_id": chunk_id,
            "source_type": "reference",
            "title": title,
            "source": str(chunk["source"]),
            "topic": str(chunk.get("topic", "")),
            "graph_nodes": [str(node_id) for node_id in graph_nodes],
            "tier": str(chunk.get("tier", "")),
            "reference_file": _relative_to_repo(source_path),
        },
    }


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _as_langchain_documents(rows: list[ReportDocument]) -> list[Any]:
    from langchain_core.documents import Document

    return [
        Document(page_content=row["page_content"], metadata=row["metadata"])
        for row in rows
    ]


def corpus_fingerprint_path(chroma_storage: str | Path) -> Path:
    """Sidecar path for the corpus fingerprint next to Chroma storage."""
    return Path(chroma_storage) / "corpus_fingerprint.json"


def compute_corpus_fingerprint(
    *,
    source_path: str | Path,
    split: str,
    report_docs: list[ReportDocument],
    reference_docs: list[ReportDocument],
    reference_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Fingerprint covering source path, split, and report/reference document contents.

    Used to refuse silently reusing a Chroma index built for a different corpus.
    """
    ref_dir = Path(reference_dir) if reference_dir is not None else get_reference_dir_default()
    payload = {
        "source_path": str(Path(source_path).resolve()),
        "split": split,
        "reference_dir": str(ref_dir.resolve()) if ref_dir.exists() else str(ref_dir),
        "report_document_count": len(report_docs),
        "reference_document_count": len(reference_docs),
        "documents": [
            {
                "page_content": doc["page_content"],
                "metadata": doc["metadata"],
            }
            for doc in report_docs + reference_docs
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "digest": digest,
        "source_path": payload["source_path"],
        "split": payload["split"],
        "reference_dir": payload["reference_dir"],
        "report_document_count": payload["report_document_count"],
        "reference_document_count": payload["reference_document_count"],
    }


def read_corpus_fingerprint(chroma_storage: str | Path) -> dict[str, Any] | None:
    path = corpus_fingerprint_path(chroma_storage)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw.get("digest"):
        return None
    return raw


def write_corpus_fingerprint(chroma_storage: str | Path, fingerprint: dict[str, Any]) -> None:
    path = corpus_fingerprint_path(chroma_storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")


def write_hybridrag_manifest(
    path: Path,
    *,
    source_path: Path,
    chroma_storage: Path,
    split: str,
    document_count: int,
    report_document_count: int | None = None,
    reference_document_count: int = 0,
    reference_dir: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "chroma_storage": str(chroma_storage.resolve()),
        "split": split,
        "document_count": document_count,
        "report_document_count": report_document_count
        if report_document_count is not None
        else document_count - reference_document_count,
        "reference_document_count": reference_document_count,
    }
    if reference_dir is not None:
        payload["reference_dir"] = str(reference_dir.resolve())
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_hybridrag_manifest(path: Path | None = None) -> dict[str, Any] | None:
    manifest_path = path or DEFAULT_MANIFEST_PATH
    if not manifest_path.exists():
        return None
    raw = json.loads(manifest_path.read_text())
    if not isinstance(raw, dict):
        return None
    if not raw.get("source_path") or not raw.get("chroma_storage"):
        return None
    return raw
