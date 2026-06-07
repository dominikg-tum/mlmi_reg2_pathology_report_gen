"""Shared helpers for offline vision scripts."""

from __future__ import annotations

from pathlib import Path

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
