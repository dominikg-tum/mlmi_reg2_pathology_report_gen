"""HippoRAG 2 semantic memory — Full Knowledge Graph implementation."""
from __future__ import annotations

"""
HippoRAG 2 semantic memory

Implementation of HyppoRag2 semantic memory system based on
https://arxiv.org/abs/2502.14802
"""

import json
import networkx as nx
from pathlib import Path

import re
import time
from typing import Any
from openai import OpenAI
import yaml
from graph.schema import Node

# Path to repository root directory
REPO_ROOT = Path(__file__).resolve().parents[1]
# Max retries of LLM queries
MAX_RETRIES = 3
# Threshold of page rank algorithm scores
PR_THRESHOLD = 0.01


class HippoRAG2Memory:
    """
    HippoRAG2 memory component
    """

    def __init__(self, index_path: str | Path | None = None, mock_llm: bool = True):
        self.index_path = Path(index_path) if index_path else None
        # Knowledge graph that contains CoT knowledge
        self.knowledge_graph = nx.DiGraph()

        # Local development mode
        self.mock_llm = mock_llm

        if not self.mock_llm:
            # Cluster mode
            cfg = load_config()
            # Load cluster LLM data
            qwen_cfg = cfg.get("qwen", {})
            api_base = qwen_cfg.get("api_base_url", "http://localhost:8000/v1")
            api_key = qwen_cfg.get("api_key", "EMPTY")
            # Create cluster LLM model
            self.model_name = qwen_cfg.get("model_name", "qwen")
            self.client = OpenAI(base_url=api_base, api_key=api_key)

    def build_index(self, train_reports_path: str, *, split: str = "train") -> None:
        """
        Create knowledge graph from CoT chains and LLM triplet extraction.
        """
        path = Path(train_reports_path)
        if not path.exists():
            raise FileNotFoundError(f"RAG: Train CoT JSONL not found: {path}")

        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                raw = json.loads(line)

                if split and raw.get("split") != split:
                    continue

                if raw.get("extraction_status", "ok") != "ok":
                    continue

                slide_id = str(raw.get("slide_id", ""))

                chains = raw.get("chains", [])

                if not chains:
                    report = str(raw.get("report") or "").strip()
                    if report:
                        triples = self._extract_triples_with_llm(report)
                        for subject, relation, obj in triples:
                            self.knowledge_graph.add_edge(subject, obj, label=relation, source_slide=slide_id)
                    continue

                for item in chains:
                    q = item.get("question") or item.get("q", "")
                    a = item.get("answer") or item.get("a", "")

                    text_to_extract = f"Question: {q} Answer: {a}"

                    triples = self._extract_triples_with_llm(text_to_extract)

                    for subject, relation, obj in triples:
                        self.knowledge_graph.add_edge(subject, obj, label=relation, source_slide=slide_id)

        # print(self.knowledge_graph.number_of_nodes(), self.knowledge_graph.number_of_edges())

        if self.index_path:
            # Create knowledge graph and write data to file
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(self.knowledge_graph, edges="links")
            self.index_path.write_text(json.dumps(data) + "\n")

    def _extract_triples_with_llm(self, text: str):
        """
        Extracts triples from CoT chains for use in knowledge graph database.
        """

        # Return example date for test purposes in dev mode
        if self.mock_llm:
            if "Question:" in text and "Answer:" in text:
                parts = text.split("Answer:")
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip()

                return [(question, "answered_with", answer)]
            else:
                return [("Unknown_Question", "has_answer", "Unknown_Answer")]

        # System prompt to extract triples from CoT
        system_prompt = """
                You are an expert pathology information extraction system.
                Extract medical knowledge triples from the following text. 
                Return ONLY a valid JSON list of lists, where each inner list has exactly 3 strings: ["Subject", "Relation", "Object"].
                Do not output any markdown formatting or explanations.
                """
        # User prompt for LLM
        user_prompt = f"Extract triples from this CoT step:\n{text}"

        for attempt in range(MAX_RETRIES):
            try:
                # Submit prompt and wait for LLM answer
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )

                # Retrieve content of LLM response
                content = clean_json_response(response.choices[0].message.content or "")
                # Load triples from json content
                triples = json.loads(content)
                valid_triples = [tuple(t) for t in triples]

                return valid_triples

            except Exception as e:
                # Retry LLM extraction 3 times before aborting
                print(f"RAG: LLM Attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        return []

    def _load_index(self) -> None:
        """
        Check if graph index is present and load it, otherwise throw error.
        """
        # Check if graph is already created
        if self.knowledge_graph.number_of_nodes() > 0:
            return

        # Check if index path is available
        if self.index_path and self.index_path.exists():
            # Load graph from index file
            with self.index_path.open(encoding="utf-8") as f:
                data = json.load(f)
                self.knowledge_graph = nx.node_link_graph(data, edges="links")
        else:
            raise RuntimeError("RAG: Index not found. Knowledge graph cant be build.")

    def _extract_concepts_from_query(self, query: str) -> list[str]:
        """
        Extracts concepts from current agent query.
        """
        # Use keyword matching in dev mode
        if self.mock_llm:
            query_lower = query.lower()
            return [str(n) for n in self.knowledge_graph.nodes if str(n).lower() in query_lower]

        # System prompt to extract concepts from agent query
        system_prompt = """
                    You are a medical concept extraction system.
                    Extract the core medical entities (diagnoses, anatomical parts, morphological features) from the query.
                    Return ONLY a valid JSON list of strings. Do not output markdown or explanations.
                    Example: ["Endometrioid Carcinoma", "Necrosis", "Uterus"]
                    """
        for attempt in range(MAX_RETRIES):
            try:
                # Submit prompt and wait for LLM answer
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Query: {query}"},
                    ],
                    temperature=0.0,
                    max_tokens=128,
                )

                # Retrieve content of LLM response
                content = clean_json_response(response.choices[0].message.content or "")
                # Load concepts from json content
                concepts = json.loads(content)

                # Return valid concept strings
                return [c.strip() for c in concepts if isinstance(c, str)]

            except Exception as e:
                # Retry LLM extraction 3 times before aborting
                print(f"RAG: Concept Extraction Attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        return []

    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str:
        """
        Searches for the relevant concepts in knowledge graph using Personalized PageRank (PPR).
        """
        # Load knowledge graph
        self._load_index()

        if self.knowledge_graph.number_of_nodes() == 0:
            return "RAG: Knowledge graph is empty."

        # Extract concepts from query via LLM
        extracted_concepts = self._extract_concepts_from_query(query)

        # Compare concepts with knowledge graph to find start node
        start_nodes = set()
        for concept in extracted_concepts:
            for graph_node in self.knowledge_graph.nodes:
                # Compare node with concept (soft compare so relevant nodes are found)
                if concept.lower() in str(graph_node).lower() or str(graph_node).lower() in concept.lower():
                    start_nodes.add(graph_node)

        # Transform start nodes to list
        start_nodes = list(start_nodes)

        if not start_nodes:
            return f"RAG: No start nodes found for query {query}"

        # Initialize every node with 0 but start node
        personalization = {node: 0.0 for node in self.knowledge_graph.nodes}
        for sn in start_nodes:
            # Split score of start nodes evenly
            personalization[sn] = 1.0 / len(start_nodes)

        # Rund page rank algorithm on knowledge graph with personalization
        try:
            pr_scores = nx.pagerank(self.knowledge_graph, personalization=personalization, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            return "RAG: Page rank failed to converge."

        # Sort found nodes via score and remove start nodes
        ranked_nodes = sorted(pr_scores.items(), key=lambda item: item[1], reverse=True)

        # Remove nodes based on threshold and return top k results
        top_k_nodes = [n for n, score in ranked_nodes if n not in start_nodes and score > PR_THRESHOLD][:k]

        if not top_k_nodes:
            return f"RAG: Could find strong connections in graph with query {query}"

        # Format nodes
        formatted_results = set()
        for node in top_k_nodes:
            for u, v, data in self.knowledge_graph.edges(node, data=True):
                relation = data.get('label', 'related_to')
                formatted_results.add(f"Fact: [{u}] --({relation})--> [{v}]")
            for u, v, data in self.knowledge_graph.in_edges(node, data=True):
                relation = data.get('label', 'related_to')
                formatted_results.add(f"Fact: [{u}] --({relation})--> [{v}]")

        formatted_results = list(formatted_results)
        if not formatted_results:
            return f"RAG: No relevant facts found for query {query}"

        return "\n".join(formatted_results[:k * 2])


def load_config() -> dict[str, Any]:
    """Loads the main path configuration YAML file."""
    path = REPO_ROOT / "configs" / "paths.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def clean_json_response(text: str) -> str:
    """
    Return clean JSON response
    """
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return text.strip()
