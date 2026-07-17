"""Diagnostic graph node schema (loaded from JSONL)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    GLOBAL_FEATURES = "global_features"
    LOCAL_FEATURES = "local_features"
    INTEGRATION = "integration"


class NodeKind(str, Enum):
    GLOBAL = "global"
    COMPARTMENT = "compartment"
    LOCAL = "local"
    INTEGRATION = "integration"
    REPORT = "report"


class InteractionType(str, Enum):
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    BOOLEAN = "boolean"
    FREE_TEXT = "free_text"


class VisualPolicy(str, Enum):
    THUMBNAIL_ONLY = "thumbnail_only"
    PATCH_RETRIEVE = "patch_retrieve"
    BOTH = "both"


class ZoomLevel(str, Enum):
    """Objective magnification for Phase 1 patch pool selection."""

    X5 = "5x"
    X10 = "10x"
    X20 = "20x"
    X40 = "40x"


# Legacy graph JSON may still use retrieval_level band keys or 4x zoom.
_RETRIEVAL_LEVEL_TO_ZOOM = {
    "low": ZoomLevel.X5,
    "medium": ZoomLevel.X10,
    "high": ZoomLevel.X20,
    "ultra": ZoomLevel.X40,
    "4x": ZoomLevel.X5,
}


@dataclass
class Node:
    id: str
    label: str
    question: str
    tier: Tier
    node_kind: NodeKind
    interaction: InteractionType
    description: str = ""
    spatial_policy: str = ""
    options: list[str] = field(default_factory=list)
    edges: dict[str, str] = field(default_factory=dict)
    zoom_level: ZoomLevel = ZoomLevel.X20
    visual_policy: VisualPolicy = VisualPolicy.PATCH_RETRIEVE
    requires_visual_evidence: bool = True
    is_leaf: bool = False
    root: bool = False

    @property
    def mag_band(self) -> str:
        """Canonical zoom key for offline cache lookup (5x/10x/20x/40x)."""
        return self.zoom_level.value

    @property
    def retrieval_text(self) -> str:
        """CONCH text-encoder input: question + optional description."""
        q = self.question.strip()
        d = self.description.strip()
        return f"{q} {d}".strip() if d else q

    def retrieval_text_with_context(
        self,
        prior_steps: list[tuple[str, str]] | None = None,
        *,
        sub_query: str = "",
    ) -> str:
        """CONCH query with optional chain summary and ReAct sub_query.

        Used at inference for spatial nodes and ReAct re-retrieve:
        ``encode_text(node.retrieval_text_with_context(steps, sub_query=I_t))``
        """
        parts = [self.retrieval_text]
        if prior_steps:
            summary = "; ".join(f"{nid}={ans}" for nid, ans in prior_steps[-4:])
            parts.append(f"Prior findings: {summary}.")
        sq = sub_query.strip()
        if sq:
            parts.append(sq)
        return " ".join(parts).strip()

    @property
    def retrieval_level_str(self) -> str:
        """Backward compat alias — retrievers select embeddings_{mag_band}.pt."""
        return self.mag_band

    def next_id(self, answer: str) -> str | None:
        if self.is_leaf:
            return None
        if self.interaction == InteractionType.MULTI_SELECT:
            return self.edges.get("__default__") or self.edges.get(answer)
        if self.interaction == InteractionType.FREE_TEXT:
            return self.edges.get("__default__")
        if answer not in self.edges:
            raise KeyError(
                f"Answer {answer!r} not a valid edge of node {self.id!r}. "
                f"Valid: {list(self.edges)}"
            )
        return self.edges[answer]

    def needs_patch_retrieval(self) -> bool:
        return self.visual_policy in (
            VisualPolicy.PATCH_RETRIEVE,
            VisualPolicy.BOTH,
        )
