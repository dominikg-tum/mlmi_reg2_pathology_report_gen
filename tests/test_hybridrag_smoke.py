from __future__ import annotations

import time
from pathlib import Path

import pytest

from graph.schema import (
    InteractionType,
    Node,
    NodeKind,
    Tier,
    VisualPolicy,
    ZoomLevel,
)
from memory.hybridrag import HybridRAGMemory

# Update path to match location of reports excel file
EXCEL_PATH = Path(__file__).parent.parent / "memory" / "case_reports_to_korea_collaborators.xlsx"


def make_node(
    node_id: str,
    question: str,
    tier: Tier,
    zoom_level: ZoomLevel = ZoomLevel.X10,
    node_kind: NodeKind = NodeKind.LOCAL,
) -> Node:
    return Node(
        id=node_id,
        label=node_id.replace("_", " ").title(),
        question=question,
        tier=tier,
        node_kind=node_kind,
        interaction=InteractionType.FREE_TEXT,
        zoom_level=zoom_level,
        visual_policy=VisualPolicy.PATCH_RETRIEVE,
    )


@pytest.fixture(scope="module")
def memory() -> HybridRAGMemory:
    if not EXCEL_PATH.exists():
        pytest.skip(f"Excel data file not found at '{EXCEL_PATH}' skipping integration tests")

    memory = HybridRAGMemory()
    memory.build_index(str(EXCEL_PATH), split="train")

    assert memory.ensemble_retriever is not None, (
        "Fixture setup failed: ensemble_retriever is None after build_index()"
    )
    return memory


RETRIEVAL_CASES: list[tuple[Node, str]] = [
    (
        make_node(
            "organ",
            "What organ does this biopsy originate from?",
            Tier.GLOBAL_FEATURES,
            ZoomLevel.X5,
            NodeKind.GLOBAL,
        ),
        "uterus endometrium biopsy tissue",
    ),
    (
        make_node(
            "nuclear_atypia",
            "Is there nuclear atypia present?",
            Tier.LOCAL_FEATURES,
            ZoomLevel.X40,
        ),
        "WT1 P40 expression nuclear atypia",
    ),
    (
        make_node(
            "mitotic_activity",
            "What is the mitotic activity?",
            Tier.LOCAL_FEATURES,
            ZoomLevel.X40,
        ),
        "mitotic figures per 10 HPF high power field",
    ),
    (
        make_node(
            "final_diagnosis",
            "What is the final diagnosis?",
            Tier.INTEGRATION,
            ZoomLevel.X40,
            NodeKind.INTEGRATION,
        ),
        "leiomyoma leiomyosarcoma endometrial carcinoma diagnosis",
    ),
]


def test_guard_raises() -> None:
    fresh = HybridRAGMemory()
    node = make_node("test", "test?", Tier.GLOBAL_FEATURES)
    with pytest.raises(RuntimeError, match="build_hybridrag_index"):
        fresh.retrieve(node, "test query")


@pytest.mark.parametrize("node,query", RETRIEVAL_CASES, ids=[n.id for n, _ in RETRIEVAL_CASES])
def test_retrieve_returns_results(memory: HybridRAGMemory, node: Node, query: str) -> None:
    start = time.perf_counter()
    results = memory.retrieve(node, query, k=3)
    elapsed = time.perf_counter() - start

    assert results, f"Empty result for node '{node.id}'"
    assert "Document 1" in results, f"Unexpected format for node '{node.id}'"
    assert node.id in results, f"Node ID missing from metadata for '{node.id}'"
    assert node.tier.value in results, f"Tier missing from metadata for '{node.id}'"
    assert elapsed < 5.0, f"Retrieval too slow for node '{node.id}': {elapsed:.2f}s"


def test_retrieve_low_level_returns_fewer_docs(memory: HybridRAGMemory) -> None:
    low_node = make_node("organ", "What organ?", Tier.GLOBAL_FEATURES, ZoomLevel.X5)
    high_node = make_node("atypia", "Is there atypia?", Tier.LOCAL_FEATURES, ZoomLevel.X40)

    low_result = memory.retrieve(low_node, "biopsy organ", k=4)
    high_result = memory.retrieve(high_node, "nuclear atypia", k=4)

    low_count = low_result.count("Document")
    high_count = high_result.count("Document")
    assert low_count <= high_count, (
        f"Expected LOW ({low_count} docs) <= HIGH ({high_count} docs)"
    )


def test_force_rebuild(tmp_path) -> None:
    if not EXCEL_PATH.exists():
        pytest.skip(f"Excel data file not found at '{EXCEL_PATH}'")

    isolated_memory = HybridRAGMemory(chroma_storage=str(tmp_path / "chroma_rebuild_test"))

    isolated_memory.build_index(str(EXCEL_PATH), split="train")
    assert isolated_memory.ensemble_retriever is not None

    isolated_memory.build_index(str(EXCEL_PATH), force_rebuild=True)
    assert isolated_memory.ensemble_retriever is not None

    node = make_node(
        "post_rebuild",
        "Is there any abnormality?",
        Tier.LOCAL_FEATURES,
        ZoomLevel.X10,
    )
    results = isolated_memory.retrieve(node, "abnormality tissue", k=2)

    assert results, "Retrieval broken after force_rebuild"
