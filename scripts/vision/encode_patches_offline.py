"""Offline: extract patches + TITAN encode -> embeddings_{level}.pt (P2 retrieval cache)."""

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
from vision.wsi_io import find_svs_files, slide_id_from_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode WSI patches with TITAN/CONCH.")
    parser.add_argument("--level", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--max-patches", type=int, default=0, help="0 = no cap")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    vcfg = load_vision_config()
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    model_id = args.model_id or default_titan_model(vcfg=vcfg)

    if args.slide:
        svs_files = list(data_dir.rglob(args.slide))
    else:
        svs_files = find_svs_files(data_dir, limit=args.limit)

    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    import torch

    encoder = TitanEncoder(model_id=model_id)
    ok, failed = 0, 0
    for svs_path in svs_files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / f"embeddings_{args.level}.pt"
        coord_path = out_dir / f"coords_{args.level}.pt"
        try:
            max_p = args.max_patches if args.max_patches > 0 else 0
            patches, coords, ps_lv0 = extract_patches(
                svs_path, mag_band=args.level, max_patches=max_p
            )
            if not patches:
                raise RuntimeError("no tissue patches")
            emb = encoder.encode_patches(patches, batch_size=args.batch_size)
            torch.save({"embeddings": emb, "coords": np.asarray(coords)}, emb_path)
            torch.save(np.asarray(coords, dtype=np.int64), coord_path)
            meta = {
                "slide_id": sid,
                "level": args.level,
                "n_patches": len(patches),
                "patch_size_lv0": ps_lv0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (out_dir / f"meta_{args.level}.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
            print(f"OK  {sid} {args.level} n={len(patches)} -> {emb_path}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {sid}: {exc}")
            failed += 1

    print(f"Done: {ok} ok, {failed} failed")


if __name__ == "__main__":
    main()
