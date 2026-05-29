"""Hard-coded diagnostic reasoning graph (Han's tree).

Design principle
----------------
The graph is a static data structure. Each node holds ONE clinical question and a
mapping from allowed answers to the next node id. The model never sees the graph;
it only answers the node it is given. This guarantees the agent cannot go
off-script: the controller (controller.py) owns all navigation.

Structure mirrors the project's top-to-bottom flow:

    GLOBAL        organ & procedure -> specimen type
       |
    COMPARTMENT   endometrium / myometrium / junctional zone / serosa /
       |          perimetrium uterina / mass-lesion
       |
    LOCAL         fine-grained features per compartment (the hard part)
       |
    INTEGRATION   synthesis / interpretation
       |
    REPORT        final diagnosis + report (leaf)

The node bodies below are PLACEHOLDERS with the right schema. Fill `question`,
`options`, and `edges` from Han's real graph when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeKind(str, Enum):
    GLOBAL = "global"            # organ, procedure, specimen — low-mag / metadata
    COMPARTMENT = "compartment"  # which anatomical region — gates local subtrees
    LOCAL = "local"              # fine-grained features — high-mag retrieval
    INTEGRATION = "integration"  # synthesis over accumulated memory
    REPORT = "report"            # leaf: generate final report


class AnswerType(str, Enum):
    CATEGORICAL = "categorical"  # one of `options` -> use guided_choice decoding
    BOOLEAN = "boolean"          # yes/no -> guided_choice ["yes", "no"]
    MEASUREMENT = "measurement"  # numeric (e.g. invasion depth %) -> regex-guided
    FREE_TEXT = "free_text"      # only for the report leaf


@dataclass
class Node:
    id: str
    question: str
    kind: NodeKind
    answer_type: AnswerType
    # Allowed answers = edge labels. For CATEGORICAL/BOOLEAN these are fed to the
    # model as a hard constraint (guided decoding) so it cannot answer off-graph.
    options: list[str] = field(default_factory=list)
    # answer -> next node id. Empty/"__report__" sentinel means terminal.
    edges: dict[str, str] = field(default_factory=dict)
    # "low" = thumbnail/low-mag (global), "high" = top-K high-mag patches (local).
    retrieval_level: str = "high"
    is_leaf: bool = False

    def next_id(self, answer: str) -> str | None:
        """Deterministic routing. Returns None at a leaf."""
        if self.is_leaf:
            return None
        if answer not in self.edges:
            raise KeyError(
                f"Answer {answer!r} not a valid edge of node {self.id!r}. "
                f"Valid: {list(self.edges)}"
            )
        return self.edges[answer]


ROOT_ID = "organ_procedure"

# ---------------------------------------------------------------------------
# TODO PLACEHOLDER graph skeleton. Replace with Han's real nodes/edges.
# Kept intentionally tiny but type-correct so controller.py runs end-to-end.
# ---------------------------------------------------------------------------
GRAPH: dict[str, Node] = {
    "organ_procedure": Node(
        id="organ_procedure",
        question="What organ and procedure does this specimen come from?",
        kind=NodeKind.GLOBAL,
        answer_type=AnswerType.CATEGORICAL,
        options=["uterus_hysterectomy", "uterus_curettage", "cervix_biopsy"],
        edges={
            "uterus_hysterectomy": "specimen_type",
            "uterus_curettage": "specimen_type",
            "cervix_biopsy": "specimen_type",
        },
        retrieval_level="low",
    ),
    "specimen_type": Node(
        id="specimen_type",
        question="What is the specimen type / tissue extent?",
        kind=NodeKind.GLOBAL,
        answer_type=AnswerType.CATEGORICAL,
        options=["full_uterus", "fragments", "endometrial_only"],
        edges={
            "full_uterus": "compartment",
            "fragments": "compartment",
            "endometrial_only": "compartment",
        },
        retrieval_level="low",
    ),
    "compartment": Node(
        id="compartment",
        question="Which compartment shows the principal lesion?",
        kind=NodeKind.COMPARTMENT,
        answer_type=AnswerType.CATEGORICAL,
        options=[
            "endometrium",
            "myometrium",
            "junctional_zone",
            "serosa",
            "perimetrium",
            "mass_lesion",
        ],
        # Each compartment routes into its own LOCAL subtree. Placeholder: all -> one.
        edges={
            "endometrium": "local_endometrium_architecture",
            "myometrium": "local_endometrium_architecture",
            "junctional_zone": "local_endometrium_architecture",
            "serosa": "local_endometrium_architecture",
            "perimetrium": "local_endometrium_architecture",
            "mass_lesion": "local_endometrium_architecture",
        },
        retrieval_level="low",
    ),
    "local_endometrium_architecture": Node(
        id="local_endometrium_architecture",
        question="What is the glandular/architectural pattern?",
        kind=NodeKind.LOCAL,
        answer_type=AnswerType.CATEGORICAL,
        options=["glandular", "papillary", "solid", "not_assessable"],
        edges={
            "glandular": "local_myometrial_invasion",
            "papillary": "local_myometrial_invasion",
            "solid": "local_myometrial_invasion",
            "not_assessable": "integration",
        },
        retrieval_level="high",
    ),
    "local_myometrial_invasion": Node(
        id="local_myometrial_invasion",
        question="Is there myometrial invasion?",
        kind=NodeKind.LOCAL,
        answer_type=AnswerType.BOOLEAN,
        options=["yes", "no"],
        edges={"yes": "integration", "no": "integration"},
        retrieval_level="high",
    ),
    "integration": Node(
        id="integration",
        question="Synthesize the findings into a final diagnosis.",
        kind=NodeKind.INTEGRATION,
        answer_type=AnswerType.FREE_TEXT,
        edges={"__report__": "report"},
        retrieval_level="high",
    ),
    "report": Node(
        id="report",
        question="Generate the final structured pathology report.",
        kind=NodeKind.REPORT,
        answer_type=AnswerType.FREE_TEXT,
        is_leaf=True,
        retrieval_level="high",
    ),
}


def validate_graph(graph: dict[str, Node] = GRAPH) -> None:
    """Sanity-check edge targets exist and leaves terminate. Call in tests."""
    for node in graph.values():
        for ans, target in node.edges.items():
            if target not in graph:
                raise ValueError(f"{node.id}: edge {ans!r} -> missing node {target!r}")
        if node.is_leaf and node.edges:
            raise ValueError(f"leaf {node.id} must have no edges")
        if node.answer_type in (AnswerType.CATEGORICAL, AnswerType.BOOLEAN):
            missing = set(node.options) - set(node.edges)
            if missing and not node.is_leaf:
                raise ValueError(f"{node.id}: options without edges: {missing}")
