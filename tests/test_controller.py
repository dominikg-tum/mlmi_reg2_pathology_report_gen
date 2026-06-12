from dataclasses import dataclass

from agent.answers import normalize_answer
from agent.backends import DummyBackend, ZeroShotQwenBackend
from agent.controller import traverse
from agent.memory import JsonGraphStore
from graph import GRAPH, ROOT_ID
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
