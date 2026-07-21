"""CLI: verify everything needed to build the LoRA dataset is present.

Checks, for every usable train slide in chains.jsonl, that the Phase-1 offline
artifacts the dataset builder depends on actually exist:
  * chains.jsonl has ok train records
  * per-slide CONCH patch pool (patch_embeddings_20x.pt)
  * per-slide thumbnail
  * resolvable .svs under the WSI data dir

Light (file stat only) — no GPU / TITAN needed. Run before submitting
scripts/cluster/build_training_jsonl.sh so you don't fail a job minutes in.

    python -m scripts.training.check_prereqs --split train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from baselines.agent_runner import load_paths_config, load_vision_cache_root
from vision.cache import build_slide_cache
from vision.mag_config import fixed_retrieval_pool
from vision.wsi_mapping import canonical_slide_id, disk_filename

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAINS = REPO_ROOT / "data" / "labels" / "chains.jsonl"


def _index_wsi_files(wsi_data_dir: Path) -> dict[str, str]:
    """One walk of the WSI dir → {filename: path}. Avoids a recursive glob per slide."""
    index: dict[str, str] = {}
    if not wsi_data_dir.exists():
        return index
    for root, _dirs, files in os.walk(wsi_data_dir):
        for fn in files:
            if fn.lower().endswith(".svs"):
                index.setdefault(fn, os.path.join(root, fn))
    return index


def _wsi_present(slide_id: str, index: dict[str, str]) -> bool:
    # slide_id may be a comma-separated multi-slide case; inference uses the first.
    for part in slide_id.split(","):
        part = part.strip()
        if part and part in index:
            return True
    return False


def _train_slides(chains_path: Path, split: str) -> list[str]:
    slides: list[str] = []
    with chains_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("extraction_status", "ok") != "ok":
                continue
            if split and str(rec.get("split", "")) != split:
                continue
            if rec.get("slide_id") and (rec.get("chain-of-thought") or []):
                slides.append(rec["slide_id"])
    return slides


def _print_missing(label: str, items: list[str], show: int) -> None:
    if not items:
        return
    print(f"  missing {label}: {len(items)}")
    for s in items[:show]:
        print(f"    - {s}")
    if len(items) > show:
        print(f"    ... and {len(items) - show} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--show", type=int, default=10, help="Max slides to list per issue")
    args = parser.parse_args()

    cfg = load_paths_config()
    cache_root = load_vision_cache_root()
    wsi_data_dir = Path(cfg["cluster"]["data_dir"])
    pool = fixed_retrieval_pool()

    print(f"chains       : {args.chains}")
    print(f"cache_root   : {cache_root}")
    print(f"wsi_data_dir : {wsi_data_dir}")
    print(f"pool level   : {pool}")

    if not args.chains.exists():
        print("\nFAIL: chains.jsonl not found. Run scripts/cluster/build_chains.sh first.")
        raise SystemExit(1)
    if cache_root is None:
        print("\nFAIL: cache_root unset (configs/vision.yaml). Cannot locate offline caches.")
        raise SystemExit(1)

    slides = _train_slides(args.chains, args.split)
    print(f"\n{args.split} slides in chains.jsonl: {len(slides)}")
    if not slides:
        print("FAIL: no usable train records (check split / extraction_status).")
        raise SystemExit(1)

    print("indexing WSI files (one pass)...", flush=True)
    wsi_index = _index_wsi_files(wsi_data_dir)
    print(f"  found {len(wsi_index)} .svs files", flush=True)

    no_thumb: list[str] = []
    no_patches: list[str] = []
    no_wsi: list[str] = []
    ready = 0

    for slide_id in slides:
        first = slide_id.split(",")[0].strip()
        canonical = canonical_slide_id(first)
        sc = build_slide_cache(cache_root, canonical)
        thumb_ok = sc.thumbnail_path is not None and sc.thumbnail_path.exists()
        pe = sc.patch_embeddings_path(pool)
        patches_ok = pe is not None and pe.exists()
        wsi_ok = _wsi_present(disk_filename(first), wsi_index) or _wsi_present(
            first, wsi_index
        )

        if not thumb_ok:
            no_thumb.append(slide_id)
        if not patches_ok:
            no_patches.append(slide_id)
        if not wsi_ok:
            no_wsi.append(slide_id)
        if thumb_ok and patches_ok and wsi_ok:
            ready += 1

    print(f"\nfully ready  : {ready}/{len(slides)}")
    _print_missing(f"patch_embeddings_{pool}.pt", no_patches, args.show)
    _print_missing("thumbnail", no_thumb, args.show)
    _print_missing(".svs file", no_wsi, args.show)

    if ready == 0:
        print("\nFAIL: no slide has all artifacts — run offline preprocessing first.")
        raise SystemExit(1)
    if ready < len(slides):
        print(
            "\nWARN: some slides are incomplete; the builder will attach thumbnails only "
            "(or skip patch retrieval) for those. Encode them for full multimodal parity."
        )
    else:
        print("\nOK: all train slides have thumbnail + patch pool + WSI. Ready to build.")


if __name__ == "__main__":
    main()
