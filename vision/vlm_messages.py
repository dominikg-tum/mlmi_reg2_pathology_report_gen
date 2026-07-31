"""Build OpenAI-compatible multimodal message parts for Qwen3-VL."""

from __future__ import annotations

import base64
import io
from pathlib import Path


def _image_part(path: Path) -> dict:
    """Encode an on-disk image for the OpenAI multimodal API.

    WSI/patch PNGs often embed multi-MB ICC profiles. Sending those bytes raw
    makes vLLM's PIL raise ``Decompressed data too large for
    PngImagePlugin.MAX_TEXT_CHUNK``. Re-encode as RGB JPEG without metadata.
    """
    try:
        from PIL import Image, PngImagePlugin

        PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90, optimize=True)
            raw = buf.getvalue()
        mime = "jpeg"
    except Exception:
        # Fallback: original bytes (may still fail on the server for heavy PNGs).
        raw = path.read_bytes()
        suffix = path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
    b64 = base64.standard_b64encode(raw).decode("ascii")
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
