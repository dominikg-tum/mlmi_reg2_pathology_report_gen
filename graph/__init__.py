"""Diagnostic reasoning graph: hard-coded structure + deterministic traversal.

The graph and controller OWN traversal. The model only answers one node's
question at a time and never decides where to go.
"""

from graph.diagnostic_graph import GRAPH, ROOT_ID, AnswerType, Node, NodeKind
from graph.controller import AnswerBackend, DummyBackend, traverse

__all__ = [
    "GRAPH",
    "ROOT_ID",
    "Node",
    "NodeKind",
    "AnswerType",
    "AnswerBackend",
    "DummyBackend",
    "traverse",
]
