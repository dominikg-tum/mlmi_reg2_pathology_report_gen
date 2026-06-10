from graph.loader import load_graph, validate_graph
from graph.schema import NodeKind, VisualPolicy, ZoomLevel


def test_load_seed_graph():
    graph, root = load_graph()
    assert root == "organ_procedure"
    assert len(graph) == 3
    validate_graph(graph)


def test_defaults_on_seed_nodes():
    graph, _ = load_graph()
    assert graph["organ_procedure"].visual_policy == VisualPolicy.THUMBNAIL_ONLY
    assert graph["organ_procedure"].zoom_level == ZoomLevel.X5
    assert graph["compartment"].zoom_level == ZoomLevel.X10
    assert graph["report"].zoom_level == ZoomLevel.X20
    assert graph["report"].visual_policy == VisualPolicy.BOTH
    assert graph["report"].is_leaf
