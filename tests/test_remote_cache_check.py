"""Tests for scripts/local/remote_cache_check.py (no cluster SSH)."""

from __future__ import annotations

from pathlib import Path

from scripts.local import remote_cache_check as rcc
from vision.cache import slide_cache_dir


def _write_required(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in rcc.REQUIRED:
        (cache_dir / name).write_bytes(b"x")


def test_incomplete_in_range_uses_manifest(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "configs").mkdir()
    (repo / "vision").mkdir()
    (repo / "scripts" / "vision").mkdir(parents=True)

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    slide_ids = ["TUM_Uterus_0001.svs", "TUM_Uterus_0002.svs", "TUM_Uterus_0003.svs"]
    (cache_root / rcc.MANIFEST_NAME).write_text("\n".join(slide_ids) + "\n")

    (repo / "configs" / "vision.yaml").write_text(f"cache_root: {cache_root}\n")
    (repo / "configs" / "paths.yaml").write_text(f"cluster:\n  data_dir: {tmp_path / 'data'}\n")

    monkeypatch.setattr(
        "vision.mag_config.VISION_CONFIG_PATH", repo / "configs" / "vision.yaml"
    )
    from vision.mag_config import load_vision_config

    load_vision_config.cache_clear()

    _write_required(slide_cache_dir(cache_root, "TUM_Uterus_0001.svs"))
    _write_required(slide_cache_dir(cache_root, "TUM_Uterus_0003.svs"))

    assert rcc.first_incomplete(repo) == 1
    assert rcc.incomplete_in_range(repo, 0, 2) == [1]
    assert rcc.slide_complete(repo, 0) is True
    assert rcc.slide_complete(repo, 1) is False
