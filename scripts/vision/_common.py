"""Shared helpers for offline vision scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from vision.mag_config import encode_levels, load_vision_config, retrieval_config
from vision.wsi_io import resolve_wsi_files

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths_config() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def default_data_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_paths_config()
    return Path(cfg["cluster"]["data_dir"])


def default_cache_root(vcfg: dict | None = None) -> Path:
    vcfg = vcfg or load_vision_config()
    return Path(vcfg["cache_root"]).expanduser()


def default_titan_model(cfg: dict | None = None, vcfg: dict | None = None) -> str:
    cfg = cfg or load_paths_config()
    vcfg = vcfg or load_vision_config()
    return vcfg.get("titan", {}).get("model_id") or cfg.get("models", {}).get(
        "titan", "MahmoodLab/TITAN"
    )


def default_encode_levels() -> list[str]:
    return encode_levels()


def default_retrieval_config() -> dict:
    return retrieval_config()


def tiling_verified_flag(cache_root: Path) -> Path:
    return cache_root / "tiling_verified.flag"


def slide_log_path(cache_root: Path, slide_id: str, suffix: str) -> Path:
    safe = slide_id.replace(",", "_").replace("/", "_")
    return cache_root / "logs" / f"{safe}.{suffix}"


def load_coords_from_pt(coord_path: Path) -> list[tuple[int, int]]:
    """Load level-0 patch coords saved by tile_slides (torch int64 [N, 2])."""
    import torch

    raw = torch.load(coord_path, map_location="cpu", weights_only=False)
    return [(int(x), int(y)) for x, y in np.asarray(raw)]


def resolve_offline_svs_files(
    data_dir: Path,
    *,
    slide: str = "",
    limit: int = 0,
    wsi_index: int | None = None,
) -> list[Path]:
    """Resolve .svs list for offline jobs (single slide, index, or batch)."""
    return resolve_wsi_files(
        data_dir, slide=slide, limit=limit, wsi_index=wsi_index
    )
