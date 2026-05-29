"""Graph integrity + controller smoke tests (no model required)."""

from graph.controller import DummyBackend, traverse
from graph.diagnostic_graph import GRAPH, ROOT_ID, validate_graph


def test_graph_edges_valid():
    validate_graph(GRAPH)


def test_root_exists():
    assert ROOT_ID in GRAPH


def test_traversal_reaches_leaf():
    chain = traverse(DummyBackend())
    assert chain, "traversal produced no steps"
    last = GRAPH[chain[-1].node_id]
    assert last.is_leaf, "traversal did not terminate at a leaf"
