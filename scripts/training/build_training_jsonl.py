"""CLI: build the Phase-1 LoRA SFT dataset from WP3 chains.

Recreates the exact Phase-1 visual pathway (thumbnail + CONCH top-k patches) per
node, so run it on the cluster where the offline caches, WSIs and TITAN/CONCH are
available. Uses the TITAN transformers pin (4.46) — NOT the training env.

    python -m scripts.training.build_training_jsonl \
        --output training/samples.jsonl --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.agent_runner import (
    load_paths_config,
    load_vision_cache_root,
    resolve_search_all_patches,
)
from retrieval.base import get_retriever
from training.dataset import build_training_jsonl
from vision.cache import build_slide_cache
from vision.thumbnail import _resolve_wsi_path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAINS = REPO_ROOT / "data" / "labels" / "chains.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "training" / "samples.jsonl"


def _build_retriever(retriever_method: str, search_all_patches):
    if retriever_method in ("none", ""):
        return None
    kwargs = {}
    if retriever_method in ("titan_cosine", "graph_guided"):
        from vision.encoders.titan import TitanEncoder

        encoder = TitanEncoder()
        kwargs["text_encoder"] = encoder.encode_text
        kwargs["search_all_patches"] = search_all_patches
    return get_retriever(retriever_method, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--images-out",
        type=Path,
        default=None,
        help="Dir you OWN for per-node patch crops (never the shared embeddings cache). "
        "Defaults next to --output.",
    )
    parser.add_argument("--visual", default="patch_retrieve")
    parser.add_argument("--retriever", default="graph_guided")
    parser.add_argument("--answer-format", choices=["json", "key"], default="json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="Max slides (0=all)")
    parser.add_argument("--search-all-patches", action="store_true")
    parser.add_argument(
        "--all-nodes",
        action="store_true",
        help="Include non-choice nodes too (default: single_select/boolean only)",
    )
    args = parser.parse_args()

    cfg = load_paths_config()
    cache_root = load_vision_cache_root()
    wsi_data_dir = Path(cfg["cluster"]["data_dir"])

    retriever = None
    if args.visual != "none":
        retriever = _build_retriever(
            args.retriever,
            resolve_search_all_patches(search_all_patches=args.search_all_patches),
        )

    def slide_cache_for(slide_id: str):
        if not cache_root:
            return None
        return build_slide_cache(cache_root, slide_id)

    def wsi_path_for(slide_cache):
        return _resolve_wsi_path(slide_cache, wsi_path=None, wsi_data_dir=wsi_data_dir)

    n = build_training_jsonl(
        args.chains,
        args.output,
        retriever=retriever,
        slide_cache_for=slide_cache_for,
        wsi_path_for=wsi_path_for,
        images_out_root=args.images_out,
        visual_method=args.visual,
        answer_format=args.answer_format,
        splits=(args.split,) if args.split else (),
        choice_nodes_only=not args.all_nodes,
        limit=args.limit,
    )
    print(f"Wrote {n} samples -> {args.output}")


if __name__ == "__main__":
    main()
