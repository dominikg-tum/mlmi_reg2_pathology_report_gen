"""Shared helpers for offline vision scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths_config() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def load_vision_config() -> dict:
    with (REPO_ROOT / "configs" / "vision.yaml").open() as f:
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
