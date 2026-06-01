"""LoRA fine-tune stub — DOMI implements with Patho-R1-style setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_lora(
    train_jsonl: Path,
    output_dir: Path,
    *,
    base_model: str,
    config: dict[str, Any] | None = None,
) -> Path:
    raise NotImplementedError(
        "Fine-tune VLM on ChainSample JSONL; expose as FineTunedBackend in agent/backends.py"
    )
