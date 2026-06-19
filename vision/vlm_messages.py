"""Build OpenAI-compatible multimodal message parts for Qwen3-VL."""

from __future__ import annotations

import base64
import io
from pathlib import Path

_VLM_MAX_EDGE_PX = 512


def _load_rgb_image(path: Path):
    from PIL import Image, PngImagePlugin

    # Some team thumbnail PNGs carry oversized iCCP/text chunks.
    PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
    with Image.open(path) as im:
        return im.convert("RGB")


def _image_part(path: Path) -> dict:
    from PIL import Image

    img = _load_rgb_image(path)
    w, h = img.size
    if max(w, h) > _VLM_MAX_EDGE_PX:
        scale = _VLM_MAX_EDGE_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
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
