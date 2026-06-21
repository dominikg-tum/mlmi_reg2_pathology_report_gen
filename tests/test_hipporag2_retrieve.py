"""HippoRAG2 retrieval policy tests (B1 v2: integration-only + node-aware)."""

from __future__ import annotations

import numpy as np

from graph.loader import load_graph
from memory.hipporag2 import HippoRAG2Memory, _IndexedStep, memory_retrieval_enabled


def _fake_memory(monkeypatch) -> HippoRAG2Memory:
    mem = HippoRAG2Memory()
    emb_comp = np.array([1.0, 0.0], dtype=np.float32)
    emb_mass = np.array([0.0, 1.0], dtype=np.float32)
    emb_synth = np.array([0.7, 0.7], dtype=np.float32)
    mem._steps = [
        _IndexedStep("compartment", "case1 compartment Q: Which? A: endometrium", emb_comp),
        _IndexedStep("compartment", "case2 compartment Q: Which? A: endometrium", emb_comp),
        _IndexedStep("mass_histologic_type", "case3 mass Q: Type? A: carcinoma", emb_mass),
        _IndexedStep("synthesis_interpretation", "case4 synthesis Q: Support? A: definitive", emb_synth),
        _IndexedStep("diagnosis", "case5 diagnosis Q: Category? A: benign", emb_synth),
    ]

    class _FakeModel:
        def encode(self, text: str):
            if "compartment" in text:
                return np.array([1.0, 0.0], dtype=np.float32)
            if "synthesis" in text or "diagnosis" in text:
                return np.array([0.7, 0.7], dtype=np.float32)
            return np.array([0.5, 0.5], dtype=np.float32)

    monkeypatch.setattr(mem, "_ensure_model", lambda: None)
    mem._model = _FakeModel()
    return mem


def test_memory_retrieval_enabled_only_on_integration_mcq_nodes():
    graph, _ = load_graph()
    assert memory_retrieval_enabled(graph["compartment"]) is False
    assert memory_retrieval_enabled(graph["endometrium_assessment"]) is False
    assert memory_retrieval_enabled(graph["stage_extent"]) is True
    assert memory_retrieval_enabled(graph["synthesis_interpretation"]) is True
    assert memory_retrieval_enabled(graph["diagnosis"]) is True
    assert memory_retrieval_enabled(graph["report"]) is False


def test_retrieve_skips_routing_nodes(monkeypatch):
    graph, _ = load_graph()
    mem = _fake_memory(monkeypatch)
    assert mem.retrieve(graph["compartment"], "Which compartment?", k=2) == ""
    assert mem.retrieve(graph["endometrium_assessment"], "Endometrial finding?", k=1) == ""


def test_retrieve_node_aware_at_integration(monkeypatch):
    graph, _ = load_graph()
    mem = _fake_memory(monkeypatch)
    out = mem.retrieve(graph["synthesis_interpretation"], "Integrated diagnosis?", k=1)
    assert "synthesis Q:" in out
    assert "definitive" in out
    assert "mass_histologic_type" not in out
    assert "compartment" not in out
