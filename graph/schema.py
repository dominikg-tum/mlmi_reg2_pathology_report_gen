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


class RetrievalLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Node:
    id: str
    label: str
    question: str
    tier: Tier
    node_kind: NodeKind
    interaction: InteractionType
    options: list[str] = field(default_factory=list)
    edges: dict[str, str] = field(default_factory=dict)
    retrieval_level: RetrievalLevel = RetrievalLevel.HIGH
    visual_policy: VisualPolicy = VisualPolicy.PATCH_RETRIEVE
    requires_visual_evidence: bool = True
    is_leaf: bool = False
    root: bool = False

    @property
    def retrieval_level_str(self) -> str:
        return self.retrieval_level.value

    def next_id(self, answer: str) -> str | None:
        if self.is_leaf:
            return None
        if self.interaction == InteractionType.MULTI_SELECT:
            return self.edges.get("__default__") or self.edges.get(answer)
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
