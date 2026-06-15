"""HippoRAG 2 semantic memory — Full Knowledge Graph implementation."""

from __future__ import annotations

import json
import networkx as nx
from pathlib import Path
from typing import Any

from graph.schema import Node


class HippoRAG2Memory:

    def __init__(self, index_path: str | Path | None = None):
        self.index_path = Path(index_path) if index_path else None

        self.knowledge_graph = nx.DiGraph()

    def _extract_triples_with_llm(self, text: str) -> list[tuple[str, str, str]]:

        prompt = f"""
        Extract medical knowledge triples from the following text. 
        Focus on pathology entities like diagnoses, morphological features, and biomarkers.
        Return ONLY a valid JSON list of lists, where each inner list has exactly 3 strings: ["Subject", "Relation", "Object"].
        Do not output any markdown formatting, just the raw JSON array.
        
        Text: {text}
        """

        # ==========================================
        # TODO: HIER KOMMT EUER ECHTER QWEN-AUFRUF HIN
        # llm_output = generate_with_qwen(prompt)
        # ==========================================

        # --- DUMMY OUTPUT ZUM TESTEN ---
        # Damit das Skript jetzt schon durchläuft, simulieren wir die Qwen-Antwort:
        llm_output = '[["Endometrioid Carcinoma", "has_feature", "Necrosis"], ["Necrosis", "indicates", "Poor Prognosis"]]'

        try:
            triples = json.loads(llm_output)
            return triples
        except json.JSONDecodeError as exception:
            raise ValueError(
                f"RAG: LLM output JSON could not be decoded: {exception}"
            ) from exception

    def build_index(self, train_reports_path: str, *, split: str = "train") -> None:
        path = Path(train_reports_path)
        if not path.exists():
            raise FileNotFoundError(f"Train CoT JSONL not found: {path}")

        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                raw = json.loads(line)

                if split and raw.get("split") not in (split, "", None):
                    if raw.get("split") != split:
                        continue

                slide_id = raw.get("slide_id", "")

                for item in raw.get("chain-of-thought") or raw.get("qa_chain") or []:
                    q = item.get("question") or item.get("q", "")
                    a = item.get("answer") or item.get("a", "")

                    text_to_extract = f"Question: {q} Answer: {a}"

                    triples = self._extract_triples_with_llm(text_to_extract)

                    for subject, relation, obj in triples:
                        self.knowledge_graph.add_edge(subject, obj, label=relation, source_slide=slide_id)

        print(self.knowledge_graph.number_of_nodes(), self.knowledge_graph.number_of_edges())

        if self.index_path:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(self.knowledge_graph)
            self.index_path.write_text(json.dumps(data) + "\n")
