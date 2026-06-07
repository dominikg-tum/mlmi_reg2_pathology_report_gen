"""Visual gate: montage preview for one slide before CONCH encoding."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.vision._common import (
    default_cache_root,
    load_vision_config,
    tiling_verified_flag,
)
from vision.cache import slide_cache_dir
from vision.patching import extract_patch_coords, load_patches_from_coords
from vision.wsi_io import slide_id_from_path


def _spread_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return np.linspace(0, n - 1, num=k, dtype=int).tolist()


def _montage(images: list[Image.Image], *, cols: int = 4) -> Image.Image:
    if not images:
        raise ValueError("no images for montage")
    w, h = images[0].size
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(img.resize((w, h)), (c * w, r * h))
    return canvas


def verify_slide(
    svs_path: Path,
    cache_root: Path,
    *,
    level: str = "high",
    n_preview: int = 16,
    write_flag: bool = True,
) -> Path:
    sid = slide_id_from_path(svs_path)
    out_dir = slide_cache_dir(cache_root, sid)
    coord_path = out_dir / f"coords_{level}.pt"
    meta_path = out_dir / f"meta_{level}.json"

    if coord_path.exists():
        coords = torch.load(coord_path, map_location="cpu", weights_only=False)
        coords = [tuple(int(x), int(y)) for x, y in np.asarray(coords)]
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    else:
        coords, patch_size_lv0 = extract_patch_coords(svs_path, mag_band=level)
        if not coords:
            raise RuntimeError("no tissue patches")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(np.asarray(coords, dtype=np.int64), coord_path)
        meta = {
            "slide_id": sid,
            "level": level,
            "n_patches": len(coords),
            "patch_size_lv0": patch_size_lv0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    indices = _spread_indices(len(coords), n_preview)
    preview_coords = [coords[i] for i in indices]
    patches = load_patches_from_coords(svs_path, preview_coords, mag_band=level)
    montage = _montage(patches)
    preview_path = out_dir / f"tiling_preview_{level}.png"
    montage.save(preview_path)

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    print(f"slide: {sid}")
    print(f"level: {level}  n_patches: {len(coords)}")
    print(f"coord x range: [{min(xs)}, {max(xs)}]  y range: [{min(ys)}, {max(ys)}]")
    print(f"patch_size_lv0: {meta.get('patch_size_lv0', '?')}")
    print(f"preview: {preview_path}")

    if write_flag:
        flag = tiling_verified_flag(cache_root)
        flag.write_text(
            json.dumps(
                {
                    "slide_id": sid,
                    "level": level,
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                    "preview": str(preview_path),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"gate flag: {flag}")

    return preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify tiling on one slide (visual gate).")
    parser.add_argument("--wsi-path", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--level", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--n-preview", type=int, default=16)
    parser.add_argument("--no-flag", action="store_true", help="Do not write tiling_verified.flag")
    args = parser.parse_args()

    vcfg = load_vision_config()
    cache_root = args.cache_root or default_cache_root(vcfg)
    verify_slide(
        args.wsi_path,
        cache_root,
        level=args.level,
        n_preview=args.n_preview,
        write_flag=not args.no_flag,
    )


if __name__ == "__main__":
    main()
