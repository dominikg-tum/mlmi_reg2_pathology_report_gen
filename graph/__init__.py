"""Diagnostic reasoning graph: JSONL structure + validation."""

from graph.loader import load_graph, validate_graph
from graph.schema import (
    InteractionType,
    Node,
    NodeKind,
    RetrievalLevel,
    Tier,
    VisualPolicy,
)

GRAPH, ROOT_ID = load_graph()

# Legacy alias used by guided-decoding backends
AnswerType = InteractionType

__all__ = [
    "GRAPH",
    "ROOT_ID",
    "Node",
    "NodeKind",
    "InteractionType",
    "AnswerType",
    "Tier",
    "VisualPolicy",
    "RetrievalLevel",
    "load_graph",
    "validate_graph",
]
