"""Build OpenAI-compatible multimodal message parts for Qwen3-VL."""

from __future__ import annotations

import base64
from pathlib import Path


def _image_part(path: Path) -> dict:
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
    }


def build_user_content(
    prompt: str,
    image_paths: list[Path] | None = None,
) -> str | list[dict]:
    """Return plain text or multimodal content list for chat.completions."""
    paths = [p for p in (image_paths or []) if p is not None and p.exists()]
    if not paths:
        return prompt
    parts: list[dict] = [_image_part(p) for p in paths]
    parts.append({"type": "text", "text": prompt})
    return parts
