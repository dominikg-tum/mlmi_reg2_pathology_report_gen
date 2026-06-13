"""Offline: build kmeans_centroids_{level}.pt from embeddings_{level}.pt (CPU only)."""

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
    slide_log_path,
)
from retrieval.kmeans_index import build_kmeans_index
from vision.cache import SlideCache, slide_cache_dir
from vision.wsi_io import resolve_wsi_files, slide_id_from_path


def _build_one(cache_root: Path, slide_id: str, out_dir: Path, level: str, k: int) -> None:
    cache = SlideCache(slide_id=slide_id, cache_dir=out_dir)
    emb_path = cache.patch_embeddings_path(level)
    centroid_path = out_dir / f"kmeans_centroids_{level}.pt"
    if centroid_path.exists():
        print(f"SKIP {out_dir.name} {level} (centroids exist)")
        return
    if emb_path is None or not emb_path.exists():
        raise FileNotFoundError(f"missing patch embeddings for {level!r}")

    data = torch.load(emb_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        emb = data.get("embeddings", data.get("emb"))
    else:
        emb = data
    emb = np.asarray(emb, dtype=np.float32)
    indices, labels = build_kmeans_index(emb, k=k)
    torch.save(indices, centroid_path)
    meta = {
        "level": level,
        "k": int(len(indices)),
        "n_patches": int(emb.shape[0]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"kmeans_meta_{level}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"OK  {out_dir.name} {level} k={len(indices)} -> {centroid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build K-means centroid indices per slide.")
    parser.add_argument("--level", choices=["5x", "10x", "20x", "40x"], action="append")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--wsi-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rcfg = default_retrieval_config()
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root()
    levels = args.level or default_encode_levels()
    k = args.k or int(rcfg.get("kmeans_k", 100))

    svs_files = resolve_wsi_files(
        data_dir, slide=args.slide, limit=args.limit, wsi_index=args.wsi_index
    )
    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    (cache_root / "logs").mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    for svs_path in svs_files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        for level in levels:
            done_path = slide_log_path(cache_root, sid, f"kmeans_{level}.done")
            fail_path = slide_log_path(cache_root, sid, f"kmeans_{level}.failed")
            try:
                _build_one(cache_root, sid, out_dir, level, k)
                done_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")
                if fail_path.exists():
                    fail_path.unlink()
                ok += 1
            except Exception:
                fail_path.write_text(traceback.format_exc())
                print(f"FAIL {sid} {level}: see {fail_path}")
                failed += 1

    print(f"Done: {ok} ok, {failed} failed")


if __name__ == "__main__":
    main()
