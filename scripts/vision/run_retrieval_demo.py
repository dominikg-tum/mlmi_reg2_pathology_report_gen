"""Smoke test: question-conditioned retrieval with adjacent-scale ancestor pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from graph.schema import Tier
from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
from vision.cache import build_slide_cache, slide_cache_dir
from vision.encoders.titan import TitanEncoder
from vision.wsi_io import slide_id_from_path
from retrieval.titan_cosine import TitanCosineRetriever

TIER_TO_LEVEL = {
    Tier.GLOBAL_FEATURES.value: "5x",
    Tier.LOCAL_FEATURES.value: "20x",
    Tier.INTEGRATION.value: "20x",
    "global_features": "5x",
    "local_features": "20x",
    "integration": "20x",
}


def _montage(pairs: list, out_path: Path) -> None:
    tiles: list[Image.Image] = []
    for rp in pairs:
        if rp.patch_image is not None:
            tiles.append(rp.patch_image)
        if rp.parent_image is not None:
            tiles.append(rp.parent_image)
        if rp.grandparent_image is not None:
            tiles.append(rp.grandparent_image)
    if not tiles:
        return
    w, h = tiles[0].size
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, img in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(img.resize((w, h)), (c * w, r * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval demo with TITAN cosine + K-means.")
    parser.add_argument("--wsi-path", type=Path, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--node-tier", type=str, default="local_features")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Simulate integration/report node (attach grandparent per config)",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    vcfg = load_vision_config()
    cache_root = args.cache_root or default_cache_root(vcfg)
    out_dir = args.output_dir or cache_root
    sid = slide_id_from_path(args.wsi_path)
    slide_cache = build_slide_cache(cache_root, sid)

    level = TIER_TO_LEVEL.get(args.node_tier, "20x")
    tier = "integration" if args.integration else args.node_tier
    node_kind = "report" if args.integration else "local"
    encoder = TitanEncoder()
    retriever = TitanCosineRetriever(text_encoder=encoder.encode_text)

    pairs = retriever.retrieve(
        args.question,
        slide_cache,
        level=level,
        k=args.k,
        wsi_path=args.wsi_path,
        return_images=True,
        tier=tier,
        node_kind=node_kind,
    )

    for i, rp in enumerate(pairs):
        parent = rp.parent_coord if rp.parent_coord else "—"
        gp = rp.grandparent_coord if rp.grandparent_coord else "—"
        print(
            f"[{i}] sim={rp.similarity:.4f} level={rp.level} "
            f"coord={rp.coord} parent={parent} ({rp.parent_level}) "
            f"grandparent={gp} ({rp.grandparent_level}) idx={rp.index}"
        )

    demo_path = out_dir / f"demo_{sid.replace('.svs', '')}.png"
    _montage(pairs, demo_path)
    print(f"montage: {demo_path}")


if __name__ == "__main__":
    main()
