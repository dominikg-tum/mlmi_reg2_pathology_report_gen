"""HippoRAG 2 semantic memory — full KG: NICK; embedding fallback for smoke tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from graph.schema import Node


@dataclass
class _IndexedStep:
    node_id: str
    text: str
    embedding: np.ndarray


class HippoRAG2Memory:
    """Lightweight sentence-transformer fallback until full HippoRAG 2 KG is wired."""

    def __init__(self, index_path: str | Path | None = None):
        self.index_path = Path(index_path) if index_path else None
        self._steps: list[_IndexedStep] = []
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_index(self, train_reports_path: str, *, split: str = "train") -> None:
        """Index CoT steps from JSONL (one record per line with chain-of-thought)."""
        self._ensure_model()
        path = Path(train_reports_path)
        if not path.exists():
            raise FileNotFoundError(f"Train CoT JSONL not found: {path}")

        steps: list[_IndexedStep] = []
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
                    nid = item.get("node_id", "")
                    q = item.get("question") or item.get("q", "")
                    a = item.get("answer") or item.get("a", "")
                    text = f"{slide_id} {nid} Q: {q} A: {a}".strip()
                    emb = self._model.encode(text)
                    steps.append(_IndexedStep(node_id=nid, text=text, embedding=np.asarray(emb)))

        self._steps = steps
        if self.index_path:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "steps": [
                    {"node_id": s.node_id, "text": s.text, "embedding": s.embedding.tolist()}
                    for s in steps
                ]
            }
            self.index_path.write_text(json.dumps(payload) + "\n")

    def load_index(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self._steps = [
            _IndexedStep(
                node_id=item["node_id"],
                text=item["text"],
                embedding=np.asarray(item["embedding"], dtype=np.float32),
            )
            for item in data.get("steps", [])
        ]

    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str:
        if not self._steps:
            if self.index_path and self.index_path.exists():
                self.load_index(self.index_path)
            else:
                return ""

        self._ensure_model()
        q_emb = np.asarray(self._model.encode(f"{node.id} {query}"), dtype=np.float32)
        sims = []
        for step in self._steps:
            denom = np.linalg.norm(q_emb) * np.linalg.norm(step.embedding) + 1e-8
            sim = float(np.dot(q_emb, step.embedding) / denom)
            sims.append((sim, step))
        sims.sort(key=lambda x: -x[0])
        top = sims[:k]
        if not top:
            return ""
        lines = [f"- {s.text} (sim={sim:.3f})" for sim, s in top]
        return "\n".join(lines)

    def online_update(self, node_id: str, question: str, answer: str) -> None:
        """Append current step to in-memory index after each graph node."""
        self._ensure_model()
        text = f"{node_id} Q: {question} A: {answer}"
        emb = self._model.encode(text)
        self._steps.append(_IndexedStep(node_id=node_id, text=text, embedding=np.asarray(emb)))
