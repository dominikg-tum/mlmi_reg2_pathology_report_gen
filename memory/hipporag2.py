from __future__ import annotations

"""
HippoRAG 2 semantic memory

Implementation of HyppoRag2 semantic memory system based on
https://arxiv.org/abs/2502.14802
"""

import json
import igraph as ig
from pathlib import Path

import re
import time
from typing import Any
from openai import OpenAI
import yaml
from graph.schema import Node
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to repository root directory
REPO_ROOT = Path(__file__).resolve().parents[1]
# Max retries of LLM queries
MAX_RETRIES = 3


class HippoRAG2Memory:
    """
    HippoRAG2 memory component
    """

    def __init__(self, index_path: str | Path | None = None, mock_llm: bool = True):
        self.index_path = Path(index_path) if index_path else None
        # Knowledge graph that contains CoT knowledge
        self.knowledge_graph = None

        self.mock_llm = mock_llm

        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        if not self.mock_llm:
            cfg = load_config()
            # Load cluster LLM data
            qwen_cfg = cfg.get("qwen", {})
            api_base = qwen_cfg.get("api_base_url", "http://localhost:8000/v1")
            api_key = qwen_cfg.get("api_key", "EMPTY")
            # Create cluster LLM model
            self.client = OpenAI(base_url=api_base, api_key=api_key)
            self.model_name = qwen_cfg.get("model_name", "qwen")

    def build_index(self, train_reports_path: str, *, split: str = "train") -> None:
        """
        Create knowledge graph from CoT chains and LLM triplet extraction.
        """
        path = Path(train_reports_path)
        if not path.exists():
            raise FileNotFoundError(f"RAG: Train CoT JSONL not found: {path}")

        passages = []

        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
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
                        passages.append({"id": slide_id, "text": report})
                else:
                    for i, item in enumerate(chains):
                        q = item.get("question") or item.get("q", "")
                        a = item.get("answer") or item.get("a", "")
                        text = f"Question: {q} Answer: {a}"
                        passages.append({"id": f"{slide_id}_chain_{i}", "text": text})

        all_triples = []
        for p_idx, p in enumerate(passages):
            triples = self._extract_triples_with_llm(p["text"])
            for sub, rel, obj in triples:
                all_triples.append({"sub": sub, "rel": rel, "obj": obj, "p_idx": p_idx})

        phrase_list = []
        phrase_to_idx = {}
        for t in all_triples:
            for ph in (t["sub"], t["obj"]):
                if ph not in phrase_to_idx:
                    phrase_to_idx[ph] = len(phrase_list)
                    phrase_list.append(ph)

        n_ph = len(phrase_list)
        n_pa = len(passages)

        phrase_embs = self._embed(phrase_list)
        synonym_pairs = self._detect_synonyms(phrase_embs, threshold=0.8)

        self.phrase_embs = phrase_embs

        edges = []
        e_types = []
        e_labels = []

        def add_edge(src, dst, etype, label):
            edges.append((src, dst))
            e_types.append(etype)
            e_labels.append(label)

        for i, j in synonym_pairs:
            add_edge(i, j, "synonym", "synonym")
            add_edge(j, i, "synonym", "synonym")

        for t in all_triples:
            s_idx = phrase_to_idx[t["sub"]]
            o_idx = phrase_to_idx[t["obj"]]
            if s_idx != o_idx:
                add_edge(s_idx, o_idx, "relation", t["rel"])
                add_edge(o_idx, s_idx, "relation", f"inv_{t['rel']}")
        pa_to_ph = {}
        for t in all_triples:
            pa_to_ph.setdefault(t["p_idx"], set()).add(phrase_to_idx[t["sub"]])
            pa_to_ph.setdefault(t["p_idx"], set()).add(phrase_to_idx[t["obj"]])

        for p_idx, ph_set in pa_to_ph.items():
            ig_p = n_ph + p_idx
            for ph_idx in ph_set:
                add_edge(ig_p, ph_idx, "context", "contains")
                add_edge(ph_idx, ig_p, "context", "is_in")

        g = ig.Graph(n=n_ph + n_pa, directed=True)
        if edges:
            g.add_edges(edges)
            g.es["type"] = e_types
            g.es["label"] = e_labels

        g.vs["type"] = ["phrase"] * n_ph + ["passage"] * n_pa
        g.vs["text"] = phrase_list + [p["text"] for p in passages]
        g.vs["original_id"] = [""] * n_ph + [p["id"] for p in passages]

        self.knowledge_graph = g
        self.phrase_list = phrase_list
        self.passages = passages
        self.phrase_to_idx = phrase_to_idx

        self.passage_embs = self._embed([p["text"] for p in passages])
        self.triples = [{"sub": t["sub"], "rel": t["rel"], "obj": t["obj"]} for t in all_triples]
        self.triple_embs = self._embed([f"{t['sub']} | {t['rel']} | {t['obj']}" for t in self.triples])

        if self.index_path:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            graph_data = {
                "n_nodes": g.vcount(),
                "edges": [(e.source, e.target, e["type"], e["label"]) for e in g.es],
                "node_types": g.vs["type"],
                "node_texts": g.vs["text"],
                "node_original_ids": g.vs["original_id"]
            }
            self.index_path.write_text(json.dumps(graph_data) + "\n")

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        return self.encoder.encode(
            texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

    def _detect_synonyms(self, phrase_embs: np.ndarray, threshold: float = 0.8) -> list[tuple[int, int]]:
        n = len(phrase_embs)
        pairs = []
        if n == 0: return pairs

        BATCH = 512
        for start in range(0, n, BATCH):
            chunk = phrase_embs[start: start + BATCH]
            sims = chunk @ phrase_embs.T
            for local_i, row in enumerate(sims):
                gi = start + local_i
                for j in range(gi + 1, n):
                    if float(row[j]) >= threshold:
                        pairs.append((gi, j))
        return pairs

    def _extract_triples_with_llm(self, text: str):
        """
        Extracts triples from CoT chains for use in knowledge graph database.
        """

        # Return example data for test purposes in dev mode
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
        if self.knowledge_graph is not None and self.knowledge_graph.vcount() > 0:
            return

        if self.index_path and self.index_path.exists():
            with self.index_path.open(encoding="utf-8") as f:
                data = json.load(f)

            n_nodes = data["n_nodes"]
            self.knowledge_graph = ig.Graph(n=n_nodes, directed=True)

            if data["edges"]:
                self.knowledge_graph.add_edges([(e[0], e[1]) for e in data["edges"]])
                self.knowledge_graph.es["type"] = [e[2] for e in data["edges"]]
                self.knowledge_graph.es["label"] = [e[3] for e in data["edges"]]

            self.knowledge_graph.vs["type"] = data["node_types"]
            self.knowledge_graph.vs["text"] = data["node_texts"]
            self.knowledge_graph.vs["original_id"] = data.get("node_original_ids", [""] * n_nodes)

            self.phrase_list = [t for i, t in enumerate(data["node_texts"]) if data["node_types"][i] == "phrase"]
            self.passages = [{"id": orig, "text": txt} for i, (orig, txt) in
                             enumerate(zip(self.knowledge_graph.vs["original_id"], data["node_texts"])) if
                             data["node_types"][i] == "passage"]
            self.phrase_to_idx = {ph: i for i, ph in enumerate(self.phrase_list)}
        else:
            raise RuntimeError("RAG: Index not found. Run build_index() beforehand.")

    def _filter_triples_llm(self, query: str, candidates: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        if self.mock_llm:
            return candidates

        candidates_json = json.dumps(
            {"fact": [[t["sub"], t["rel"], t["obj"]] for t, _ in candidates]},
            ensure_ascii=False
        )

        system_prompt = """
        You are a critical component of a high-stakes question-answering system.
        Select up to 4 relevant facts from the candidate list that have a strong connection to the query.
        Output ONLY valid JSON: {"fact": [["s1","p1","o1"], ["s2","p2","o2"]]}
        If no facts are relevant, return {"fact": []}.
        """
        user_prompt = f"Question: {query}\nFact Before Filter: {candidates_json}\nFact After Filter:"

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\s*```$", "", raw).strip()

                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    data = json.loads(raw)

                kept = {tuple(str(x) for x in item)
                        for item in data.get("fact", []) if len(item) == 3}
                return [(t, score) for t, score in candidates
                        if (t["sub"], t["rel"], t["obj"]) in kept]
            except Exception as e:
                print(f"RAG: Filter Attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        return candidates

    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str:
        self._load_index()

        if self.knowledge_graph.vcount() == 0:
            return "RAG: Knowledge graph is empty."

        q_emb = self._embed([query])[0]

        if len(self.triples) == 0:
            return "RAG: No triples found in graph."

        sims = self.triple_embs @ q_emb
        top_k_triples = min(5, len(self.triples))
        top_idx = np.argsort(sims)[::-1][:top_k_triples]
        candidates = [(self.triples[i], float(sims[i])) for i in top_idx]

        filtered_triples = self._filter_triples_llm(query, candidates) if candidates else []

        if not filtered_triples:
            pa_sims = self.passage_embs @ q_emb
            top_idx = np.argsort(pa_sims)[::-1][:k]
            results = []
            for i in top_idx:
                p = self.passages[i]
                results.append(f"[Source: {p['id']} | Dense-Score: {pa_sims[i]:.4f}]\n{p['text']}")
            return "\n\n---\n\n".join(results)

        p = np.zeros(self.knowledge_graph.vcount(), dtype=np.float64)

        ph_score_lists = {}
        for t, sim in filtered_triples:
            for phrase in (t["sub"], t["obj"]):
                idx = self.phrase_to_idx.get(phrase, -1)
                if idx >= 0:
                    ph_score_lists.setdefault(idx, []).append(sim)

        ranked_ph = sorted([(idx, float(np.mean(scores))) for idx, scores in ph_score_lists.items()],
                           key=lambda x: x[1], reverse=True)[:5]
        for ph_idx, score in ranked_ph:
            p[ph_idx] = score

        n_ph = len(self.phrase_list)
        pa_sims = self.passage_embs @ q_emb
        for pa_idx, sim in enumerate(pa_sims):
            p[n_ph + pa_idx] = float(sim) * 0.05

        total = p.sum()
        if total > 0:
            p /= total
        else:
            p[:] = 1.0 / len(p)

        scores = self.knowledge_graph.personalized_pagerank(
            vertices=None, directed=True, damping=0.5, reset=p.tolist()
        )

        ranked_passages = []
        for pa_idx in range(len(self.passages)):
            ig_idx = n_ph + pa_idx
            ranked_passages.append((self.passages[pa_idx], scores[ig_idx]))

        ranked_passages.sort(key=lambda x: x[1], reverse=True)

        formatted_results = []
        for passage, score in ranked_passages[:k]:
            source_info = f"Source: {passage['id']}" if passage['id'] else "Source: Unknown"
            formatted_results.append(f"[{source_info} | PPR-Score: {score:.4f}]\n{passage['text']}")

        return "\n\n---\n\n".join(formatted_results)


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
