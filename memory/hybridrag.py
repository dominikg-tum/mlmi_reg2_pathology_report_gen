"""
HybridRAG semantic memory
- RAG for getting familiar and testing purposes
- Semantical vector search and "BM25" (https://en.wikipedia.org/wiki/Okapi_BM25)
"""

from __future__ import annotations

import gc
import os
import re
import shutil
import time

from graph.schema import Node, RetrievalLevel
import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from memory.base import SemanticMemory

# Storage file for Semantic Embeddings (needs to get generalized)
CHROMA_STORAGE_DEFAULT  = "./chroma_db_storage"


class HybridRAGMemory(SemanticMemory):

    def __init__(self, chroma_storage: str = CHROMA_STORAGE_DEFAULT):
        self.chroma_storage = chroma_storage
        self.vector_storage = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.vector_retriever = None

        # Initialise embedding model:
        # NeuML/pubmedbert-base-embeddings or BAAI/bge-m3 (more powerful but bigger) for pathology
        self.embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

    def build_index(self, train_reports_path: str, *, split: str = "train", force_rebuild: bool = False) -> None:
        # Read reports file with pandas
        try:
            df = pd.read_excel(train_reports_path)
        except FileNotFoundError as exception:
            raise FileNotFoundError(
                f"RAG: Reports Excel not found at '{train_reports_path}'"
            ) from exception

        # Optional: Only use reports with case_class A and B as others are not complete
        df = df[df["case_class"].isin(['A', 'B'])]
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
            weights=[0.6, 0.4]
        )

    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str:
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
    return re.findall(r'\w+', text.lower())
