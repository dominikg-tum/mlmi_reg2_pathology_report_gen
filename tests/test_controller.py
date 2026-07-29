from dataclasses import dataclass

from agent.answers import normalize_answer
from agent.backends import DummyBackend, ZeroShotQwenBackend
from agent.controller import traverse
from agent.memory import JsonGraphStore
from graph import GRAPH, ROOT_ID
from graph.schema import InteractionType, Node, NodeKind, Tier, VisualPolicy
from vision.backends import VisualBundle


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


def test_normalize_verbose_choice_answer():
    node = GRAPH["compartment"]
    assert normalize_answer("Answer: endometrium.", node) == "endometrium"
    assert normalize_answer('{"answer": "myometrium"}', node) == "myometrium"


class _RetryBackend:
    def __init__(self):
        self.calls = 0

    def answer(self, node, visual, memory, *, extra_context=""):
        self.calls += 1
        if self.calls == 1:
            return "The lesion appears endometrial.", 0.9
        return node.options[0], 0.9


def test_traversal_retries_invalid_graph_answer():
    backend = _RetryBackend()
    steps = traverse(backend, skip_report_nodes=True)
    assert steps
    assert backend.calls > len(steps)


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message
    logprobs: object | None = None


@dataclass
class _Response:
    choices: list[_Choice]


class _Completions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Response([_Choice(_Message("endometrium"))])


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_qwen_backend_sends_choices_and_images(tmp_path):
    image = tmp_path / "patch.png"
    image.write_bytes(b"not-a-real-png")
    client = _Client()
    backend = ZeroShotQwenBackend(client, "mock-model")
    node = GRAPH["compartment"]

    answer, _ = backend.answer(
        node,
        VisualBundle(patch_paths=[image], metadata={"visual": "patch_retrieve"}),
        [],
    )

    assert answer == "endometrium"
    call = client.chat.completions.kwargs
    assert call["extra_body"]["guided_choice"] == node.options
    assert call["messages"][0]["role"] == "system"
    assert isinstance(call["messages"][1]["content"], list)


def test_fixed_visual_bundle_bypasses_retrieval(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    visual = VisualBundle(
        thumbnail_path=image,
        metadata={"visual": "uploaded_image"},
    )

    steps = traverse(
        DummyBackend(),
        retriever_method="graph_guided",
        fixed_visual_bundle=visual,
    )

    assert steps[-1].node_id == "report"


class _RouteBackend:
    """Tracks whether structured JSON or plain answer path was used."""

    def __init__(self):
        self.answer_ids: list[str] = []
        self.json_ids: list[str] = []

    def answer(self, node, visual, memory, *, extra_context=""):
        self.answer_ids.append(node.id)
        if node.options:
            return node.options[0], 0.9
        return "free text report", 0.9

    def complete_json(self, node, visual, *, system_prompt, user_prompt):
        self.json_ids.append(node.id)
        key = (node.options or ["yes"])[0]
        return {"answer_key": key, "confidence": 0.9}, 0.9, "raw"


def test_structured_answer_skips_non_choice_nodes():
    """FREE_TEXT / MULTI_SELECT must use backend.answer, not Step A JSON."""
    graph = {
        "choice": Node(
            id="choice",
            label="choice",
            question="Pick one?",
            tier=Tier.GLOBAL_FEATURES,
            node_kind=NodeKind.GLOBAL,
            interaction=InteractionType.SINGLE_SELECT,
            description="",
            options=["a", "b"],
            edges={"a": "multi", "b": "multi"},
            visual_policy=VisualPolicy.THUMBNAIL_ONLY,
            requires_visual_evidence=False,
            is_leaf=False,
            root=True,
        ),
        "multi": Node(
            id="multi",
            label="multi",
            question="Select features?",
            tier=Tier.LOCAL_FEATURES,
            node_kind=NodeKind.LOCAL,
            interaction=InteractionType.MULTI_SELECT,
            description="",
            options=["x", "y"],
            edges={"x": "report", "y": "report"},
            visual_policy=VisualPolicy.THUMBNAIL_ONLY,
            requires_visual_evidence=False,
            is_leaf=False,
            root=False,
        ),
        "report": Node(
            id="report",
            label="report",
            question="Write the report.",
            tier=Tier.INTEGRATION,
            node_kind=NodeKind.REPORT,
            interaction=InteractionType.FREE_TEXT,
            description="",
            options=[],
            edges={},
            visual_policy=VisualPolicy.THUMBNAIL_ONLY,
            requires_visual_evidence=False,
            is_leaf=True,
            root=False,
        ),
    }

    backend = _RouteBackend()
    steps = traverse(
        backend,
        graph=graph,
        root_id="choice",
        structured_answer=True,
        skip_report_nodes=False,
    )

    assert [s.node_id for s in steps] == ["choice", "multi", "report"]
    assert backend.json_ids == ["choice"]
    assert backend.answer_ids == ["multi", "report"]
    assert steps[0].answer_branch == "structured"
    assert steps[1].answer_branch == "plain"
    assert steps[1].answer_branch_skip_reason == "not_choice_node"
    assert steps[2].answer_branch == "plain"


def test_invalid_react_answer_retries_on_single_call_path(monkeypatch, tmp_path):
    """A bad ReAct answer must not replay the whole A/B/C loop."""
    import sys
    import types

    from agent import node_react as node_react_mod
    from vision.cache import SlideCache

    # Stub the TITAN import so traverse can build a retriever without weights.
    fake_titan = types.ModuleType("vision.encoders.titan")
    fake_titan.TitanEncoder = lambda: types.SimpleNamespace(encode_text=lambda q: None)
    monkeypatch.setitem(sys.modules, "vision.encoders.titan", fake_titan)

    graph = {
        "patch": Node(
            id="patch",
            label="patch",
            question="Pick one?",
            tier=Tier.LOCAL_FEATURES,
            node_kind=NodeKind.LOCAL,
            interaction=InteractionType.SINGLE_SELECT,
            description="",
            options=["a", "b"],
            edges={},
            visual_policy=VisualPolicy.PATCH_RETRIEVE,
            requires_visual_evidence=True,
            is_leaf=True,
            root=True,
        ),
    }

    react_calls = []
    react_bundle = VisualBundle(metadata={"visual": "patch_retrieve", "retrieved_patches": []})

    def _fake_react(node, **kwargs):
        react_calls.append(node.id)
        return node_react_mod.NodeReactResult(
            answer_key="not-an-option",
            confidence=0.9,
            node_traces=[],
            bundle=react_bundle,
        )

    monkeypatch.setattr(node_react_mod, "run_node_react", _fake_react)

    class _Backend(_RouteBackend):
        def __init__(self):
            super().__init__()
            self.answer_bundles = []

        def answer(self, node, visual, memory, *, extra_context=""):
            self.answer_bundles.append(visual)
            return super().answer(node, visual, memory, extra_context=extra_context)

    backend = _Backend()
    steps = traverse(
        backend,
        graph=graph,
        root_id="patch",
        node_react=True,
        retriever_method="graph_guided",
        slide_cache=SlideCache(slide_id="s.svs", cache_dir=tmp_path, thumbnail_path=None),
    )

    assert [s.node_id for s in steps] == ["patch"]
    assert react_calls == ["patch"]  # ReAct ran once, not once per attempt
    assert backend.answer_ids == ["patch"]
    assert backend.answer_bundles == [react_bundle]  # fallback reuses ReAct evidence
    assert steps[0].answer_branch == "plain"
    assert steps[0].answer_branch_skip_reason == "react_invalid_answer"


def test_node_react_falls_back_without_retriever():
    """--node-react on a patch node without retriever/slide cache must not crash."""
    graph = {
        "patch": Node(
            id="patch",
            label="patch",
            question="Pick one?",
            tier=Tier.LOCAL_FEATURES,
            node_kind=NodeKind.LOCAL,
            interaction=InteractionType.SINGLE_SELECT,
            description="",
            options=["a", "b"],
            edges={},
            visual_policy=VisualPolicy.PATCH_RETRIEVE,
            requires_visual_evidence=True,
            is_leaf=True,
            root=True,
        ),
    }

    backend = _RouteBackend()
    steps = traverse(
        backend,
        graph=graph,
        root_id="patch",
        node_react=True,
        retriever_method="none",
    )

    assert [s.node_id for s in steps] == ["patch"]
    assert backend.json_ids == []
    assert backend.answer_ids == ["patch"]
    assert steps[0].answer_branch == "plain"
    assert "no_retriever" in steps[0].answer_branch_skip_reason
    assert "no_slide_cache" in steps[0].answer_branch_skip_reason
