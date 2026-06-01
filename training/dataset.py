"""Build training JSONL from WP3 chains + same visual pathway as inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ChainSample:
    slide_id: str
    node_id: str
    question: str
    target_answer: str
    visual_paths: list[str]
    episodic_context: str


def build_training_jsonl(
    chains_path: Path,
    output_path: Path,
    *,
    retriever=None,
    visual_method: str = "thumbnail",
) -> None:
    """DOMI: unroll chains into per-node samples using retriever at train time."""
    raise NotImplementedError(
        "Load WP3 chains JSONL, call retrieval per node, write training/samples.jsonl"
    )


def load_chain_samples(path: Path) -> list[ChainSample]:
    raise NotImplementedError
