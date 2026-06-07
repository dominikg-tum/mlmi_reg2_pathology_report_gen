"""Offline: CONCH encode pre-tiled coords -> embeddings_{level}.pt (P2 retrieval cache)."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.vision._common import (
    default_cache_root,
    default_data_dir,
    default_encode_levels,
    default_retrieval_config,
    default_titan_model,
    load_vision_config,
    slide_log_path,
    tiling_verified_flag,
)
from vision.cache import slide_cache_dir
from vision.encoders.titan import TitanEncoder
from vision.patching import extract_patches, load_patches_from_coords
from vision.wsi_io import resolve_wsi_files, slide_id_from_path


def _require_tiling_gate(cache_root: Path, force: bool) -> None:
    flag = tiling_verified_flag(cache_root)
    if force:
        return
    if not flag.exists():
        raise SystemExit(
            f"Refusing to encode: missing {flag}. "
            "Run scripts.vision.verify_tiling on one slide first, or pass --force-encode."
        )


def _encode_one(
    encoder: TitanEncoder,
    svs_path: Path,
    out_dir: Path,
    level: str,
    *,
    batch_size: int,
    max_patches: int,
) -> None:
    emb_path = out_dir / f"embeddings_{level}.pt"
    coord_path = out_dir / f"coords_{level}.pt"
    meta_path = out_dir / f"meta_{level}.json"

    if emb_path.exists() and coord_path.exists() and meta_path.exists():
        print(f"SKIP {slide_id_from_path(svs_path)} {level} (embeddings exist)")
        return

    if coord_path.exists():
        raw = torch.load(coord_path, map_location="cpu", weights_only=False)
        coords = [tuple(int(x), int(y)) for x, y in np.asarray(raw)]
        if max_patches > 0:
            coords = coords[:max_patches]
        patches = load_patches_from_coords(svs_path, coords, mag_band=level)
        meta_existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        patch_size_lv0 = int(meta_existing.get("patch_size_lv0", 0))
        if patch_size_lv0 <= 0:
            raise RuntimeError(f"coords exist but meta_{level}.json missing patch_size_lv0")
    else:
        max_p = max_patches if max_patches > 0 else 0
        patches, coords, patch_size_lv0 = extract_patches(
            svs_path, mag_band=level, max_patches=max_p
        )
        if not patches:
            raise RuntimeError("no tissue patches")

    emb = encoder.encode_patches(patches, batch_size=batch_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": emb, "coords": np.asarray(coords)}, emb_path)
    torch.save(np.asarray(coords, dtype=np.int64), coord_path)
    meta = {
        "slide_id": slide_id_from_path(svs_path),
        "level": level,
        "n_patches": len(patches),
        "patch_size_lv0": patch_size_lv0,
        "embedding_dim": int(emb.shape[1]) if emb.size else 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"OK  {slide_id_from_path(svs_path)} {level} n={len(patches)} -> {emb_path}")


def _check_only(cache_root: Path, data_dir: Path, levels: list[str]) -> None:
    files = resolve_wsi_files(data_dir)
    done, failed, missing = 0, 0, 0
    for svs_path in files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        for level in levels:
            done_path = slide_log_path(cache_root, sid, f"encoded_{level}.done")
            fail_path = slide_log_path(cache_root, sid, f"encoded_{level}.failed")
            emb_path = out_dir / f"embeddings_{level}.pt"
            if done_path.exists() or emb_path.exists():
                done += 1
            elif fail_path.exists():
                failed += 1
            else:
                missing += 1
    total = len(files) * len(levels)
    print(f"Encoding status: {done} done / {failed} failed / {missing} missing (of {total} tasks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode WSI patches with TITAN/CONCH.")
    parser.add_argument("--level", choices=["low", "medium", "high"], action="append")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--wsi-index", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=0, help="0 = no cap")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force-encode", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    vcfg = load_vision_config()
    rcfg = default_retrieval_config()
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    model_id = args.model_id or default_titan_model(vcfg=vcfg)
    levels = args.level or default_encode_levels()
    batch_size = args.batch_size or int(vcfg.get("titan", {}).get("batch_size", 64))

    if args.check_only:
        _check_only(cache_root, data_dir, levels)
        return

    _require_tiling_gate(cache_root, args.force_encode)

    svs_files = resolve_wsi_files(
        data_dir, slide=args.slide, limit=args.limit, wsi_index=args.wsi_index
    )
    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    encoder = TitanEncoder(model_id=model_id)
    (cache_root / "logs").mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    for svs_path in svs_files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        for level in levels:
            done_path = slide_log_path(cache_root, sid, f"encoded_{level}.done")
            fail_path = slide_log_path(cache_root, sid, f"encoded_{level}.failed")
            try:
                _encode_one(
                    encoder,
                    svs_path,
                    out_dir,
                    level,
                    batch_size=batch_size,
                    max_patches=args.max_patches,
                )
                done_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")
                if fail_path.exists():
                    fail_path.unlink()
                ok += 1
            except Exception:
                fail_path.write_text(traceback.format_exc())
                print(f"FAIL {sid} {level}: see {fail_path}")
                failed += 1

    print(f"Done: {ok} ok, {failed} failed (kmeans_k={rcfg.get('kmeans_k', 100)})")


if __name__ == "__main__":
    main()
