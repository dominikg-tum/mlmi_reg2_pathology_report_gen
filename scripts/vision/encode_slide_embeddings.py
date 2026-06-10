"""Offline: TITAN slide embedding (×20 CONCH patches only) + optional thumbnail.

TITAN was trained on 512×512 @ 20× — slide_embedding.pt always comes from the
high-magnification pool. Multi-scale CONCH pools (×4/×10/×20/×40) are encoded
separately via encode_patches_offline.py for Phase 1 zoom_level retrieval.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.vision._common import (
    default_cache_root,
    default_data_dir,
    default_titan_model,
    load_vision_config,
)
from vision.cache import slide_cache_dir
from vision.encoders.titan import TitanEncoder
from vision.patching import extract_patches
from vision.wsi_io import find_svs_files, slide_id_from_path, write_thumbnail


def _save_evidence_patches(
    patches: list,
    coords: list[tuple[int, int]],
    out_dir: Path,
    *,
    k: int = 3,
) -> list[str]:
    """Save k spread-out tissue patches as PNGs for multimodal VLM input."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not patches:
        return []
    indices = np.linspace(0, len(patches) - 1, num=min(k, len(patches)), dtype=int)
    names: list[str] = []
    for i, idx in enumerate(indices):
        name = f"patch_{i}.png"
        patches[int(idx)].save(out_dir / name)
        names.append(name)
    return names


def encode_one_slide(
    encoder: TitanEncoder,
    svs_path: Path,
    out_dir: Path,
    *,
    max_patches: int,
    batch_size: int,
    write_thumb: bool,
    max_edge_px: int,
) -> None:
    import torch

    sid = slide_id_from_path(svs_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_thumb:
        write_thumbnail(svs_path, out_dir / "thumbnail.png", max_edge_px=max_edge_px)

    patches, coords, patch_size_lv0 = extract_patches(
        svs_path,
        mag_band="20x",
        max_patches=max_patches,
    )
    if not patches:
        raise RuntimeError(f"No tissue patches extracted from {svs_path}")

    patch_emb = encoder.encode_patches(patches, batch_size=batch_size)
    slide_emb = encoder.encode_slide(
        patch_emb,
        np.asarray(coords, dtype=np.int64),
        patch_size_lv0,
    )

    torch.save(slide_emb, out_dir / "slide_embedding.pt")

    # Canonical ×20 pool artifacts — only if encode_patches_offline has not run yet.
    canonical_emb = out_dir / "patch_embeddings_20x.pt"
    canonical_coords = out_dir / "coords_20x.pt"
    if not canonical_emb.exists():
        torch.save(
            {"embeddings": patch_emb, "coords": np.asarray(coords)},
            canonical_emb,
        )
    if not canonical_coords.exists():
        torch.save(np.asarray(coords, dtype=np.int64), canonical_coords)

    evidence_names = _save_evidence_patches(
        patches, coords, out_dir / "evidence", k=3
    )

    meta = {
        "slide_id": sid,
        "source": str(svs_path),
        "model_id": encoder.model_id,
        "n_patches": len(patches),
        "patch_size_lv0": patch_size_lv0,
        "embedding_dim": int(slide_emb.shape[0]),
        "evidence_files": evidence_names,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TITAN slide embedding offline job.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-thumbnail", action="store_true")
    args = parser.parse_args()

    vcfg = load_vision_config()
    titan_cfg = vcfg.get("titan", {})
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    model_id = args.model_id or default_titan_model(vcfg=vcfg)
    max_patches = args.max_patches or int(titan_cfg.get("max_patches_per_slide", 512))
    batch_size = args.batch_size or int(titan_cfg.get("batch_size", 32))
    max_edge = int(vcfg.get("thumbnail", {}).get("max_edge_px", 1024))

    if args.slide:
        svs_files = list(data_dir.rglob(args.slide))
    else:
        svs_files = find_svs_files(data_dir, limit=args.limit)

    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    encoder = TitanEncoder(model_id=model_id)
    ok, failed = 0, 0
    for svs_path in svs_files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        try:
            encode_one_slide(
                encoder,
                svs_path,
                out_dir,
                max_patches=max_patches,
                batch_size=batch_size,
                write_thumb=not args.no_thumbnail,
                max_edge_px=max_edge,
            )
            print(f"OK  {sid} -> {out_dir / 'slide_embedding.pt'}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {sid}: {exc}")
            failed += 1

    print(f"Done: {ok} ok, {failed} failed -> {cache_root}")


if __name__ == "__main__":
    main()
