from agent.backends import DummyBackend
from agent.controller import chain_to_dict, traverse
from graph import GRAPH


def test_skip_report_nodes_stops_before_report():
    steps = traverse(DummyBackend(), skip_report_nodes=True)
    assert steps
    assert all(GRAPH[s.node_id].node_kind.value != "report" for s in steps)
    # Traversal stops on the integration node that feeds the report leaf.
    assert steps[-1].node_id == "diagnosis"
    assert GRAPH[steps[-1].node_id].edges == {
        "benign": "report",
        "premalignant": "report",
        "malignant": "report",
        "non_neoplastic": "report",
        "descriptive": "report",
    }


def test_chain_to_dict_omits_report_when_requested():
    steps = traverse(DummyBackend(), skip_report_nodes=True)
    out = chain_to_dict(steps, slide_id="CASE.svs", include_report=False)
    assert out["report"] == ""
    assert out["node_path"] == [s.node_id for s in steps]
    assert "node_id" in out["chain-of-thought"][0]
