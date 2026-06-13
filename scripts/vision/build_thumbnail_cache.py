"""Offline: .svs -> thumbnail PNG per slide (P1 baseline, no TITAN)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.vision._common import (
    default_cache_root,
    default_data_dir,
    load_vision_config,
)
from vision.cache import slide_cache_dir
from vision.wsi_io import find_svs_files, slide_id_from_path, write_thumbnail


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blurry whole-slide thumbnails.")
    parser.add_argument("--data-dir", type=Path, default=None, help="WSI root (default: paths.yaml)")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Process only N slides (0 = all)")
    parser.add_argument("--slide", type=str, default="", help="Single slide filename e.g. foo.svs")
    args = parser.parse_args()

    vcfg = load_vision_config()
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    max_edge = int(vcfg.get("thumbnail", {}).get("max_edge_px", 1024))

    if args.slide:
        svs_files = [data_dir / args.slide] if (data_dir / args.slide).exists() else []
        if not svs_files:
            svs_files = list(data_dir.rglob(args.slide))
    else:
        svs_files = find_svs_files(data_dir, limit=args.limit)

    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    ok, failed = 0, 0
    for svs_path in svs_files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        out_path = out_dir / "thumbnail.png"
        try:
            write_thumbnail(svs_path, out_path, max_edge_px=max_edge)
            meta = {
                "slide_id": sid,
                "source": str(svs_path),
                "max_edge_px": max_edge,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (out_dir / "thumbnail_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"OK  {sid} -> {out_path}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {sid}: {exc}")
            failed += 1

    print(f"Done: {ok} ok, {failed} failed -> {cache_root}")


if __name__ == "__main__":
    main()
