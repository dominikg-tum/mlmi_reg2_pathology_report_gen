"""
HybridRAG semantic memory

Baseline RAG system for pipeline tests and ablation study.
Combines semantic vector search (ChromaDB + PubMedBERT) with
lexical keyword search (BM25).
"""

from __future__ import annotations

import gc
import os
import re
import shutil
from pathlib import Path
from typing import Any
import yaml

from graph.schema import Node, RetrievalLevel
import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from memory.base import SemanticMemory

# Path to repository root directory
REPO_ROOT = Path(__file__).resolve().parents[1]


class HybridRAGMemory(SemanticMemory):
    """
    HybridRAG memory component

    Combines semantic vector search (ChromaDB + PubMedBERT) with
    lexical keyword search (BM25). Vector search captures general conceptual meaning
    of a pathology report. BM25 ensures that highly specific biomarkers
    and abbreviations (e.g., "P40", "WT1") are not missed due to vector dilution.
    """

    def __init__(self):
        """
        Initializes the HybridRAGMemory, setting up storage paths and the
        domain-specific embedding model.
        """
        self.chroma_storage = get_chroma_storage_default()
        self.vector_storage = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.vector_retriever = None

        # Initialise embedding model:
        # NeuML/pubmedbert-base-embeddings or BAAI/bge-m3 (more powerful but bigger) for pathology
        self.embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

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

        # Read reports file with pandas#
        # Correct file path management is redundant at this point
        # because of switch to CoT chains in future
        try:
            df = pd.read_excel(train_reports_path)
        except FileNotFoundError as exception:
            raise FileNotFoundError(
                f"RAG: Reports Excel not found at '{train_reports_path}'"
            ) from exception

        # Optional: Only use reports with case_class A and B as others are not complete
        # 28 of 220 cases are in class A and B so it doesn't really make sense to remove C-E
        # df = df[df["case_class"].isin(['A', 'B'])]
        # Remove empty reports if exist
        df = df.dropna(subset=['english_reports'])

        # Dropping rows that do not match split
        if 'split' in df.columns:
            df = df[df['split'] == split]

        documents = []
        # Iterate every row and create document of report with ID as metadata
        for index, row in df.iterrows():
            text = str(row['english_reports'])
            slide_id = str(row.get('slide_ids', str(index)))

            documents.append(Document(
                page_content=text,
                metadata={"slide_id": slide_id}
            ))

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

        if self.ensemble_retriever is None:
            raise RuntimeError("RAG: build_index() must be called before retrieve()")

        # Update k dynamically based on retrieval level
        effective_k = {
            RetrievalLevel.LOW: max(1, k // 2),
            RetrievalLevel.MEDIUM: k,
            RetrievalLevel.HIGH: min(k * 2, 20),
        }[node.retrieval_level]

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


def get_chroma_storage_default():
    """Retrieves the default path for Chroma database from the config."""
    cfg = load_config()
    return Path(cfg["rag"]["chroma_db_storage"])
