from graph.loader import load_graph, validate_graph
from graph.schema import InteractionType, NodeKind, VisualPolicy, ZoomLevel


def test_load_graph_root_and_validity():
    graph, root = load_graph()
    assert root == "organ_procedure"
    roots = [n for n in graph.values() if n.root]
    assert len(roots) == 1
    assert roots[0].id == "organ_procedure"
    validate_graph(graph)


def test_exactly_one_report_leaf():
    graph, _ = load_graph()
    leaves = [n for n in graph.values() if n.is_leaf]
    assert len(leaves) == 1
    report = leaves[0]
    assert report.id == "report"
    assert report.node_kind == NodeKind.REPORT
    assert not report.edges


def test_global_and_compartment_defaults():
    graph, _ = load_graph()
    assert graph["organ_procedure"].zoom_level == ZoomLevel.X5
    assert graph["organ_procedure"].visual_policy == VisualPolicy.THUMBNAIL_ONLY
    assert graph["compartment"].zoom_level == ZoomLevel.X10
    assert graph["compartment"].visual_policy == VisualPolicy.THUMBNAIL_ONLY


def test_compartment_does_not_jump_to_report():
    """The compartment node must route into a local work-up, never straight to report."""
    graph, _ = load_graph()
    assert "report" not in graph["compartment"].edges.values()
    assert graph["compartment"].edges["endometrium"] == "endometrium_assessment"


def test_no_40x_zoom_levels():
    """40x is skipped offline; nuclei/mitosis nodes use 20x instead."""
    graph, _ = load_graph()
    assert all(node.zoom_level != ZoomLevel.X40 for node in graph.values())


def test_zoom_and_visual_policy_by_kind():
    graph, _ = load_graph()
    for node in graph.values():
        if node.node_kind == NodeKind.LOCAL:
            assert node.zoom_level == ZoomLevel.X20
            assert node.visual_policy == VisualPolicy.PATCH_RETRIEVE
        if node.node_kind in (NodeKind.INTEGRATION, NodeKind.REPORT):
            assert node.zoom_level == ZoomLevel.X20
            assert node.visual_policy == VisualPolicy.BOTH


def test_every_select_option_has_matching_edge():
    graph, _ = load_graph()
    for node in graph.values():
        if node.is_leaf:
            continue
        if node.interaction in (
            InteractionType.SINGLE_SELECT,
            InteractionType.BOOLEAN,
        ):
            assert set(node.options) <= set(node.edges)


def test_full_endometrium_carcinoma_path_reaches_report():
    """The endometrial carcinoma path should walk through staging to the report leaf."""
    graph, _ = load_graph()
    chain = [
        ("compartment", "endometrium", "endometrium_assessment"),
        ("endometrium_assessment", "carcinoma", "endometrial_carcinoma_subtype"),
        ("endometrial_carcinoma_subtype", "endometrioid", "endometrial_carcinoma_grade"),
        ("endometrial_carcinoma_grade", "grade_1", "background_endometrium"),
        ("background_endometrium", "atrophic_background", "stage_extent"),
        ("stage_extent", "deep_invasion", "synthesis_interpretation"),
        ("synthesis_interpretation", "definitive", "diagnosis"),
        ("diagnosis", "malignant", "report"),
    ]
    for node_id, answer, expected in chain:
        assert graph[node_id].next_id(answer) == expected
    assert graph["report"].is_leaf
