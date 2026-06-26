"""Export likely lesion patches at 5x using PathAgent-style PLIP retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
from vision.cache import slide_cache_dir
from vision.wsi_io import iter_tissue_patches, resolve_wsi_files, slide_id_from_path

DEFAULT_LESION_QUERY = (
    "lesion, neoplasm, tumor, dysplasia, atypia, abnormal gland, abnormal tissue, "
    "pathologic finding"
)


class PatchScorer(Protocol):
    def score(self, query: str, patch_images: list, *, batch_size: int = 16) -> np.ndarray: ...


@dataclass
class PathAgentPlipScorer:
    """PathAgent Navigator-style text/image scoring through PLIP."""

    pathagent_root: Path
    plip_lib_path: Path
    plip_ckpt: Path

    def __post_init__(self) -> None:
        self.pathagent_root = Path(self.pathagent_root).expanduser()
        self.plip_lib_path = Path(self.plip_lib_path).expanduser()
        self.plip_ckpt = Path(self.plip_ckpt).expanduser()
        for path in (self.pathagent_root, self.plip_lib_path):
            if not path.exists():
                raise FileNotFoundError(path)
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        try:
            from plip import PLIP

            self._plip = PLIP(str(self.plip_ckpt))
            self._backend = "pathagent_plip"
        except ModuleNotFoundError:
            self._plip = _TransformersPlip(self.plip_ckpt)
            self._backend = "transformers_plip"

    def score(self, query: str, patch_images: list, *, batch_size: int = 16) -> np.ndarray:
        text = _as_numpy(self._plip.encode_text([query], batch_size=1))
        image = _as_numpy(self._plip.encode_images(patch_images, batch_size=batch_size))
        text = _l2_normalize(text)
        image = _l2_normalize(image)
        return (image @ text.T).reshape(-1)


class _TransformersPlip:
    """Minimal PLIP/CLIP wrapper for local Hugging Face model folders."""

    def __init__(self, model_path: Path | str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.model_path = str(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        self.model = CLIPModel.from_pretrained(self.model_path).to(self.device).eval()

    def encode_text(self, texts: list[str], *, batch_size: int = 1):
        import torch

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs.append(self.model.get_text_features(**inputs).detach().cpu())
        return torch.cat(outputs, dim=0).numpy()

    def encode_images(self, images: list, *, batch_size: int = 16):
        import torch

        outputs = []
        for start in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs.append(self.model.get_image_features(**inputs).detach().cpu())
        return torch.cat(outputs, dim=0).numpy()


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(value, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return value / denom


def _safe_coord_name(coord: tuple[int, int]) -> str:
    return f"{int(coord[0])}_{int(coord[1])}"


def extract_lesion_patches_5x(
    *,
    svs_path: Path,
    cache_root: Path,
    scorer: PatchScorer,
    query: str = DEFAULT_LESION_QUERY,
    top_k: int = 1,
    patch_size: int = 224,
    max_candidates: int = 0,
    batch_size: int = 16,
    background_threshold: int = 220,
) -> dict:
    """Tile one WSI at 5x, score tissue patches, and save top lesion candidates."""
    slide_id = slide_id_from_path(svs_path)
    out_dir = slide_cache_dir(cache_root, slide_id)
    patch_dir = out_dir / "lesion_patches_5x"
    patch_dir.mkdir(parents=True, exist_ok=True)

    patches = []
    coords: list[tuple[int, int]] = []
    patch_size_lv0 = 0
    for image, coord, ps_lv0 in iter_tissue_patches(
        svs_path,
        objective="5x",
        patch_size=patch_size,
        stride=patch_size,
        background_threshold=background_threshold,
        max_patches=max_candidates,
    ):
        patches.append(image)
        coords.append(coord)
        patch_size_lv0 = int(ps_lv0)

    if not patches:
        raise RuntimeError(f"No tissue patches found at 5x for {svs_path}")

    scores = scorer.score(query, patches, batch_size=batch_size)
    if scores.shape[0] != len(patches):
        raise RuntimeError(
            f"Scorer returned {scores.shape[0]} scores for {len(patches)} patches"
        )

    order = np.argsort(scores)[::-1][: max(1, int(top_k))]
    selected = []
    for rank, index in enumerate(order, start=1):
        coord = coords[int(index)]
        out_path = patch_dir / f"lesion_rank{rank:03d}_{_safe_coord_name(coord)}.png"
        patches[int(index)].save(out_path)
        selected.append(
            {
                "rank": rank,
                "coord": [int(coord[0]), int(coord[1])],
                "score": float(scores[int(index)]),
                "patch_path": str(out_path),
            }
        )

    manifest = {
        "slide_id": slide_id,
        "source_wsi": str(svs_path),
        "method": "pathagent_plip_text_image_retrieval",
        "level": "5x",
        "query": query,
        "patch_size": int(patch_size),
        "patch_size_lv0": int(patch_size_lv0),
        "n_candidates": int(len(patches)),
        "selected": selected,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "lesion_patches_5x.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract likely lesion patches at 5x using PathAgent-style PLIP retrieval."
    )
    parser.add_argument("--pathagent-root", type=Path, default=Path("/Volumes/Xun/PathAgent"))
    parser.add_argument("--plip-lib-path", type=Path, required=True)
    parser.add_argument("--plip-ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--wsi-index", type=int, default=None)
    parser.add_argument("--query", type=str, default=DEFAULT_LESION_QUERY)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--background-threshold", type=int, default=220)
    args = parser.parse_args()

    vcfg = load_vision_config()
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    svs_files = resolve_wsi_files(
        data_dir,
        slide=args.slide,
        limit=args.limit,
        wsi_index=args.wsi_index,
    )
    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    scorer = PathAgentPlipScorer(
        pathagent_root=args.pathagent_root,
        plip_lib_path=args.plip_lib_path,
        plip_ckpt=args.plip_ckpt,
    )

    ok, failed = 0, 0
    for svs_path in svs_files:
        try:
            manifest = extract_lesion_patches_5x(
                svs_path=svs_path,
                cache_root=cache_root,
                scorer=scorer,
                query=args.query,
                top_k=args.top_k,
                patch_size=args.patch_size,
                max_candidates=args.max_candidates,
                batch_size=args.batch_size,
                background_threshold=args.background_threshold,
            )
            print(json.dumps(manifest, indent=2))
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
