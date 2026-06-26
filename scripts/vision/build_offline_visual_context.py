"""Build offline visual context for graph-guided VLM inference.

For each WSI this runs:
1. PathAgent-style PLIP retrieval to export likely 5x lesion patches.
2. UNI2 patch/slide embedding extraction at selected magnifications.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
from scripts.vision.encode_uni2_wsi import encode_slide_with_uni2
from scripts.vision.extract_pathagent_lesion_patches import (
    DEFAULT_LESION_QUERY,
    PathAgentPlipScorer,
    extract_lesion_patches_5x,
)
from vision.encoders.uni2 import UNI2Encoder
from vision.wsi_io import resolve_wsi_files, slide_id_from_path

DEFAULT_UNI2_LEVELS = ("1.25x", "2.5x", "5x", "10x")


def build_visual_context_for_slide(
    *,
    svs_path: Path,
    cache_root: Path,
    lesion_scorer,
    uni2_encoder,
    lesion_query: str = DEFAULT_LESION_QUERY,
    lesion_top_k: int = 1,
    lesion_patch_size: int = 224,
    lesion_max_candidates: int = 0,
    lesion_batch_size: int = 16,
    levels: list[str] | None = None,
    uni2_patch_size: int = 224,
    uni2_batch_size: int = 16,
    uni2_max_patches: int = 0,
    save_uni2_patch_images: bool = False,
    background_threshold: int = 220,
) -> dict:
    lesion = extract_lesion_patches_5x(
        svs_path=svs_path,
        cache_root=cache_root,
        scorer=lesion_scorer,
        query=lesion_query,
        top_k=lesion_top_k,
        patch_size=lesion_patch_size,
        max_candidates=lesion_max_candidates,
        batch_size=lesion_batch_size,
        background_threshold=background_threshold,
    )
    uni2 = encode_slide_with_uni2(
        svs_path=svs_path,
        cache_root=cache_root,
        encoder=uni2_encoder,
        levels=levels or list(DEFAULT_UNI2_LEVELS),
        patch_size=uni2_patch_size,
        batch_size=uni2_batch_size,
        max_patches=uni2_max_patches,
        background_threshold=background_threshold,
        save_patch_images=save_uni2_patch_images,
    )
    return {
        "slide_id": slide_id_from_path(svs_path),
        "source_wsi": str(svs_path),
        "lesion_patches": lesion,
        "uni2": uni2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build lesion-patch and UNI2 embedding caches for WSIs."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--wsi-index", type=int, default=None)

    parser.add_argument("--pathagent-root", type=Path, default=Path("/Volumes/Xun/PathAgent"))
    parser.add_argument("--plip-lib-path", type=Path, required=True)
    parser.add_argument("--plip-ckpt", type=Path, required=True)
    parser.add_argument("--lesion-query", type=str, default=DEFAULT_LESION_QUERY)
    parser.add_argument("--lesion-top-k", type=int, default=1)
    parser.add_argument("--lesion-patch-size", type=int, default=224)
    parser.add_argument("--lesion-max-candidates", type=int, default=0)
    parser.add_argument("--lesion-batch-size", type=int, default=16)

    parser.add_argument("--uni-repo-path", type=Path, default=None)
    parser.add_argument("--uni2-weights-path", type=Path, default=None)
    parser.add_argument("--uni-model-name", default=None)
    parser.add_argument("--level", action="append", choices=list(DEFAULT_UNI2_LEVELS))
    parser.add_argument("--uni2-patch-size", type=int, default=None)
    parser.add_argument("--uni2-batch-size", type=int, default=None)
    parser.add_argument("--uni2-max-patches", type=int, default=None)
    parser.add_argument("--save-uni2-patch-images", action="store_true")
    parser.add_argument("--background-threshold", type=int, default=220)
    args = parser.parse_args()

    vcfg = load_vision_config()
    uni_cfg = vcfg.get("uni2", {})
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    uni_repo_path = args.uni_repo_path or Path(uni_cfg.get("repo_path", "/Volumes/Xun/UNI"))
    uni2_weights_path = args.uni2_weights_path or uni_cfg.get("weights_path")
    uni_model_name = args.uni_model_name or str(uni_cfg.get("model_name", "uni2-h"))
    levels = args.level or list(uni_cfg.get("levels", DEFAULT_UNI2_LEVELS))
    uni2_patch_size = args.uni2_patch_size or int(uni_cfg.get("patch_size", 224))
    uni2_batch_size = args.uni2_batch_size or int(uni_cfg.get("batch_size", 16))
    uni2_max_patches = (
        args.uni2_max_patches
        if args.uni2_max_patches is not None
        else int(uni_cfg.get("max_patches_per_level", 0))
    )
    save_uni2_patch_images = bool(
        args.save_uni2_patch_images or uni_cfg.get("save_patch_images", False)
    )

    svs_files = resolve_wsi_files(
        data_dir,
        slide=args.slide,
        limit=args.limit,
        wsi_index=args.wsi_index,
    )
    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    lesion_scorer = PathAgentPlipScorer(
        pathagent_root=args.pathagent_root,
        plip_lib_path=args.plip_lib_path,
        plip_ckpt=args.plip_ckpt,
    )
    uni2_encoder = UNI2Encoder(
        repo_path=uni_repo_path,
        model_name=uni_model_name,
        weights_path=uni2_weights_path,
    )

    ok, failed = 0, 0
    for svs_path in svs_files:
        try:
            result = build_visual_context_for_slide(
                svs_path=svs_path,
                cache_root=cache_root,
                lesion_scorer=lesion_scorer,
                uni2_encoder=uni2_encoder,
                lesion_query=args.lesion_query,
                lesion_top_k=args.lesion_top_k,
                lesion_patch_size=args.lesion_patch_size,
                lesion_max_candidates=args.lesion_max_candidates,
                lesion_batch_size=args.lesion_batch_size,
                levels=levels,
                uni2_patch_size=uni2_patch_size,
                uni2_batch_size=uni2_batch_size,
                uni2_max_patches=uni2_max_patches,
                save_uni2_patch_images=save_uni2_patch_images,
                background_threshold=args.background_threshold,
            )
            print(json.dumps(result, indent=2))
            ok += 1
        except Exception:
            failed += 1
            print(f"FAIL {slide_id_from_path(svs_path)}")
            print(traceback.format_exc())
    print(f"Done: {ok} ok, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
