"""
HybridRAG semantic memory

Baseline RAG system for pipeline tests and ablation study.
Combines semantic vector search (ChromaDB + PubMedBERT) with
lexical keyword search (BM25).
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
import yaml

from graph.schema import Node
import pandas as pd

from extraction.labels_io import assign_splits
from memory.base import SemanticMemory

# Path to repository root directory
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "memory" / "hybridrag_manifest.json"
ReportDocument = dict[str, Any]


class HybridRAGMemory(SemanticMemory):
    """
    HybridRAG memory component

    Combines semantic vector search (ChromaDB + PubMedBERT) with
    lexical keyword search (BM25). Vector search captures general conceptual meaning
    of a pathology report. BM25 ensures that highly specific biomarkers
    and abbreviations (e.g., "P40", "WT1") are not missed due to vector dilution.
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

    def build_index(self, train_reports_path: str, *, split: str = "train", force_rebuild: bool = False) -> None:
        """
        Builds or loads the ensemble index containing both Vector and BM25 databases.

        Reads the pathology reports, filters them by dataset split and converts
        them into Document objects. Chroma vector database is
        saved to disk and reloaded if it already exists, unless force_rebuild is True.

        Args:
            train_reports_path (str): Path to the Excel file containing the reports.
            split (str): Which dataset split to index (default: "train").
            force_rebuild (bool): If True, deletes any existing Chroma database
                                  on disk and rebuilds embeddings.

        Raises:
            FileNotFoundError: If the specified Excel file does not exist.
        """

        documents = _as_langchain_documents(
            load_report_documents(train_reports_path, split=split)
        )
        from langchain_chroma import Chroma
        from langchain_community.retrievers import BM25Retriever
        from langchain_classic.retrievers import EnsembleRetriever

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

        # Load semantic embeddings storage from file
        if os.path.exists(self.chroma_storage):
            self.vector_storage = Chroma(
                persist_directory=self.chroma_storage,
                embedding_function=self.embeddings
            )
        # Create semantic embeddings and save to file
        else:
            self.vector_storage = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=self.chroma_storage
            )

        # Create retriever in initialization progress
        self.vector_retriever = self.vector_storage.as_retriever()

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=clean_tokenize)

        # Combining semantic and BM25 retriever into ensemble
        # Could also do ablation study on weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.65, 0.35]
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
                self.build_index(
                    str(source_path),
                    split=str(manifest.get("split", "train")),
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
            node (Node): Current node in the diagnostic graph.
            query (str): Raw search string.
            k (int): Base number of documents to retrieve.

        Returns:
            str: A formatted string concatenating the top-k retrieved documents,
                 including their source IDs and node metadata.

        Raises:
            RuntimeError: If called before `build_index()` is executed.
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

        # Invoke query to receive best k matches for current prompt
        results = self.ensemble_retriever.invoke(enriched_query)[:effective_k]
        formatted_results = []
        # Format result for better readability
        for i, doc in enumerate(results):
            source_id = doc.metadata.get('slide_id', 'Unknown')
            formatted_results.append(
                f"[Document {i + 1} | Source: {source_id} | Node: {node.id} | Tier: {node.tier.value}]\n"
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


def clean_tokenize(text: str) -> list[str]:
    """
    Custom tokenizer for BM25 lexical search.
    Strips punctuation and converts text to lowercase.

    Args:
        text (str): The raw text string to tokenize.

    Returns:
        list[str]: A list of cleaned, lowercase word tokens.
    """
    return re.findall(r'\w+', text.lower())


def load_config() -> dict[str, Any]:
    """Loads the main path configuration YAML file."""
    path = REPO_ROOT / "configs" / "paths.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def get_chroma_storage_default() -> Path:
    """Retrieves the default path for Chroma database from the config."""
    cfg = load_config()
    return Path(cfg["rag"]["chroma_db_storage"])


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
            {"page_content": text, "metadata": {"slide_id": slide_id}}
        )
    return documents


def _as_langchain_documents(rows: list[ReportDocument]) -> list[Any]:
    from langchain_core.documents import Document

    return [
        Document(page_content=row["page_content"], metadata=row["metadata"])
        for row in rows
    ]


def write_hybridrag_manifest(
    path: Path,
    *,
    source_path: Path,
    chroma_storage: Path,
    split: str,
    document_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(source_path.resolve()),
        "chroma_storage": str(chroma_storage.resolve()),
        "split": split,
        "document_count": document_count,
    }
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
