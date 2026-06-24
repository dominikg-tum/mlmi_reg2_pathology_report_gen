"""Tile WSIs at selected magnifications and encode patches with UNI2."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
from vision.cache import slide_cache_dir
from vision.encoders.uni2 import UNI2Encoder, mean_pool_embeddings
from vision.wsi_io import iter_tissue_patches, resolve_wsi_files, slide_id_from_path, write_thumbnail

DEFAULT_UNI2_LEVELS = ("1.25x", "2.5x", "5x", "10x")


class PatchEncoder(Protocol):
    def encode_patches(self, patch_images: list, *, batch_size: int = 16) -> np.ndarray: ...


def _safe_level(level: str) -> str:
    return level.lower().replace("×", "x").replace(".", "p")


def _patch_dir(out_dir: Path, level: str) -> Path:
    return out_dir / "patches" / _safe_level(level)


def _encode_level(
    *,
    encoder: PatchEncoder,
    svs_path: Path,
    out_dir: Path,
    level: str,
    patch_size: int,
    batch_size: int,
    max_patches: int,
    background_threshold: int,
    save_patch_images: bool,
) -> dict:
    patches = []
    coords = []
    patch_size_lv0 = 0
    patch_paths: list[str] = []
    image_dir = _patch_dir(out_dir, level)
    if save_patch_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for index, (img, coord, ps_lv0) in enumerate(
        iter_tissue_patches(
            svs_path,
            objective=level,
            patch_size=patch_size,
            stride=patch_size,
            background_threshold=background_threshold,
            max_patches=max_patches,
        )
    ):
        patches.append(img)
        coords.append(coord)
        patch_size_lv0 = ps_lv0
        if save_patch_images:
            image_path = image_dir / f"patch_{index:06d}.png"
            img.save(image_path)
            patch_paths.append(str(image_path))

    if not patches:
        raise RuntimeError(f"no tissue patches at {level}")

    embeddings = encoder.encode_patches(patches, batch_size=batch_size)
    slide_embedding = mean_pool_embeddings(embeddings)
    coords_array = np.asarray(coords, dtype=np.int64)

    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_level(level)
    patch_emb_path = out_dir / f"uni2_patch_embeddings_{safe}.pt"
    coords_path = out_dir / f"uni2_coords_{safe}.pt"
    slide_emb_path = out_dir / f"uni2_slide_embedding_{safe}.pt"
    meta_path = out_dir / f"uni2_meta_{safe}.json"

    torch.save({"embeddings": embeddings, "coords": coords_array}, patch_emb_path)
    torch.save(coords_array, coords_path)
    torch.save(slide_embedding, slide_emb_path)

    meta = {
        "slide_id": slide_id_from_path(svs_path),
        "encoder": "uni2-h",
        "level": level,
        "patch_size": patch_size,
        "patch_size_lv0": int(patch_size_lv0),
        "n_patches": int(len(coords)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "patch_embeddings_path": str(patch_emb_path),
        "coords_path": str(coords_path),
        "slide_embedding_path": str(slide_emb_path),
        "patch_image_dir": str(image_dir) if save_patch_images else "",
        "patch_image_paths": patch_paths,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def encode_slide_with_uni2(
    *,
    svs_path: Path,
    cache_root: Path,
    encoder: PatchEncoder,
    levels: list[str],
    patch_size: int = 224,
    batch_size: int = 16,
    max_patches: int = 0,
    background_threshold: int = 220,
    save_patch_images: bool = False,
    thumbnail_max_edge_px: int = 1024,
) -> dict:
    slide_id = slide_id_from_path(svs_path)
    out_dir = slide_cache_dir(cache_root, slide_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = out_dir / "thumbnail.png"
    if not thumbnail_path.exists():
        write_thumbnail(svs_path, thumbnail_path, max_edge_px=thumbnail_max_edge_px)

    per_level = []
    for level in levels:
        per_level.append(
            _encode_level(
                encoder=encoder,
                svs_path=svs_path,
                out_dir=out_dir,
                level=level,
                patch_size=patch_size,
                batch_size=batch_size,
                max_patches=max_patches,
                background_threshold=background_threshold,
                save_patch_images=save_patch_images,
            )
        )

    summary = {
        "slide_id": slide_id,
        "thumbnail_path": str(thumbnail_path),
        "levels": per_level,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "uni2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="UNI2 WSI patching and embedding.")
    parser.add_argument("--level", action="append", choices=list(DEFAULT_UNI2_LEVELS))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--wsi-index", type=int, default=None)
    parser.add_argument("--repo-path", type=Path, default=None, help="Path to cloned UNI repo")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--background-threshold", type=int, default=220)
    parser.add_argument("--save-patch-images", action="store_true")
    args = parser.parse_args()

    vcfg = load_vision_config()
    uni_cfg = vcfg.get("uni2", {})
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    levels = args.level or list(uni_cfg.get("levels", DEFAULT_UNI2_LEVELS))
    repo_path = args.repo_path or Path(uni_cfg.get("repo_path", "/Volumes/Xun/UNI"))
    model_name = args.model_name or str(uni_cfg.get("model_name", "uni2-h"))
    patch_size = args.patch_size or int(uni_cfg.get("patch_size", 224))
    batch_size = args.batch_size or int(uni_cfg.get("batch_size", 16))
    max_patches = (
        args.max_patches
        if args.max_patches is not None
        else int(uni_cfg.get("max_patches_per_level", 0))
    )
    save_patch_images = bool(args.save_patch_images or uni_cfg.get("save_patch_images", False))

    svs_files = resolve_wsi_files(
        data_dir,
        slide=args.slide,
        limit=args.limit,
        wsi_index=args.wsi_index,
    )
    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    encoder = UNI2Encoder(repo_path=repo_path, model_name=model_name)
    ok, failed = 0, 0
    for svs_path in svs_files:
        try:
            summary = encode_slide_with_uni2(
                svs_path=svs_path,
                cache_root=cache_root,
                encoder=encoder,
                levels=levels,
                patch_size=patch_size,
                batch_size=batch_size,
                max_patches=max_patches,
                background_threshold=args.background_threshold,
                save_patch_images=save_patch_images,
            )
            print(json.dumps(summary, indent=2))
            ok += 1
        except Exception:
            failed += 1
            print(f"FAIL {slide_id_from_path(svs_path)}")
            print(traceback.format_exc())
    print(f"Done: {ok} ok, {failed} failed")


if __name__ == "__main__":
    main()

