from agent.backends import DummyBackend
from agent.controller import traverse
from agent.memory import JsonGraphStore
from graph import GRAPH, ROOT_ID


def test_traversal_reaches_leaf():
    steps = traverse(DummyBackend())
    assert steps
    assert GRAPH[steps[-1].node_id].is_leaf


def test_chain_has_next_question():
    steps = traverse(DummyBackend())
    assert steps[0].next_question
    assert not steps[-1].next_question


def test_graph_store_next():
    store = JsonGraphStore(GRAPH)
    n = store.get_node(ROOT_ID)
    nxt = store.next(n.id, n.options[0])
    assert nxt == "compartment"
