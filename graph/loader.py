"""Load and validate the execution graph from JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graph.schema import (
    InteractionType,
    Node,
    NodeKind,
    RetrievalLevel,
    Tier,
    VisualPolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = REPO_ROOT / "data" / "graph" / "execution_graph.jsonl"


def _default_retrieval_level(node_kind: NodeKind, tier: Tier | None = None) -> RetrievalLevel:
    if tier == Tier.GLOBAL_FEATURES:
        return RetrievalLevel.MEDIUM
    if tier in (Tier.LOCAL_FEATURES, Tier.INTEGRATION):
        return RetrievalLevel.HIGH
    if node_kind in (NodeKind.GLOBAL, NodeKind.COMPARTMENT):
        return RetrievalLevel.MEDIUM
    if node_kind == NodeKind.LOCAL:
        return RetrievalLevel.HIGH
    if node_kind in (NodeKind.INTEGRATION, NodeKind.REPORT):
        return RetrievalLevel.HIGH
    return RetrievalLevel.MEDIUM


def _default_visual_policy(node_kind: NodeKind) -> VisualPolicy:
    if node_kind in (NodeKind.GLOBAL, NodeKind.COMPARTMENT):
        return VisualPolicy.THUMBNAIL_ONLY
    if node_kind in (NodeKind.INTEGRATION, NodeKind.REPORT):
        return VisualPolicy.BOTH
    return VisualPolicy.PATCH_RETRIEVE


def _parse_node(raw: dict[str, Any]) -> Node:
    node_kind = NodeKind(raw["node_kind"])
    tier = Tier(raw.get("tier", Tier.GLOBAL_FEATURES.value))
    retrieval_raw = raw.get("retrieval_level")
    visual_raw = raw.get("visual_policy")
    return Node(
        id=raw["id"],
        label=raw.get("label", raw["id"]),
        question=raw["question"],
        tier=tier,
        node_kind=node_kind,
        interaction=InteractionType(raw.get("interaction", "single_select")),
        options=list(raw.get("options") or []),
        edges=dict(raw.get("edges") or {}),
        retrieval_level=RetrievalLevel(retrieval_raw)
        if retrieval_raw
        else _default_retrieval_level(node_kind, tier),
        visual_policy=VisualPolicy(visual_raw)
        if visual_raw
        else _default_visual_policy(node_kind),
        requires_visual_evidence=bool(raw.get("requires_visual_evidence", True)),
        is_leaf=bool(raw.get("is_leaf", False)),
        root=bool(raw.get("root", False)),
    )


def load_jsonl(path: Path | None = None) -> tuple[dict[str, Node], str]:
    path = path or DEFAULT_GRAPH_PATH
    graph: dict[str, Node] = {}
    root_id: str | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            node = _parse_node(raw)
            graph[node.id] = node
            if node.root:
                root_id = node.id
    if not graph:
        raise ValueError(f"No nodes loaded from {path}")
    if root_id is None:
        root_id = next(iter(graph))
    return graph, root_id


def validate_graph(graph: dict[str, Node]) -> None:
    for node in graph.values():
        for _ans, target in node.edges.items():
            if target not in graph:
                raise ValueError(
                    f"{node.id}: edge -> missing node {target!r}"
                )
        if node.is_leaf and node.edges:
            raise ValueError(f"leaf {node.id} must have no outgoing edges")
        if node.interaction in (
            InteractionType.SINGLE_SELECT,
            InteractionType.BOOLEAN,
        ):
            if not node.is_leaf:
                missing = set(node.options) - set(node.edges)
                if missing:
                    raise ValueError(
                        f"{node.id}: options without edges: {missing}"
                    )


def load_graph(path: Path | None = None) -> tuple[dict[str, Node], str]:
    graph, root_id = load_jsonl(path)
    validate_graph(graph)
    return graph, root_id
