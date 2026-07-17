"""complete_json must soft-fail on malformed VLM JSON (retry, do not abort)."""

from __future__ import annotations

from dataclasses import dataclass

from agent.backends import ZeroShotQwenBackend
from graph import GRAPH
from vision.backends import VisualBundle


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
    def __init__(self, content: str):
        self.content = content

    def create(self, **kwargs):
        return _Response([_Choice(_Message(self.content))])


class _Client:
    def __init__(self, content: str):
        self.chat = type("Chat", (), {"completions": _Completions(content)})()


def test_complete_json_malformed_returns_empty_dict():
    backend = ZeroShotQwenBackend(_Client("not json at all"), "mock")
    parsed, _conf, raw = backend.complete_json(
        GRAPH["compartment"],
        VisualBundle(),
        system_prompt="sys",
        user_prompt="user",
    )
    assert parsed == {}
    assert raw == "not json at all"


def test_complete_json_extracts_embedded_object():
    backend = ZeroShotQwenBackend(
        _Client('Sure.\n{"answer_key": "endometrium", "confidence": 0.9}\n'),
        "mock",
    )
    parsed, _conf, _raw = backend.complete_json(
        GRAPH["compartment"],
        VisualBundle(),
        system_prompt="sys",
        user_prompt="user",
    )
    assert parsed["answer_key"] == "endometrium"
