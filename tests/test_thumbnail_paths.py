"""Thumbnail path resolution from dataset banks vs per-slide cache."""

from __future__ import annotations

from pathlib import Path

from vision.cache import (
    build_slide_cache,
    dataset_thumbnail_path,
    resolve_thumbnail_path,
    slide_id_to_stem,
)


def test_slide_id_to_stem():
    assert slide_id_to_stem("TUM_Uterus_0001.svs") == "TUM_Uterus_0001"
    assert slide_id_to_stem("TUM_Uterus_0001") == "TUM_Uterus_0001"


def test_dataset_thumbnail_path_prefers_configured_variant(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "thumbnails_kmeans").mkdir(parents=True)
    jpg = root / "thumbnails_kmeans" / "TUM_Uterus_0001.jpg"
    jpg.write_bytes(b"fake")

    vcfg = {
        "thumbnail": {
            "dataset_root": str(root),
            "variant": "thumbnails_kmeans",
        }
    }
    assert dataset_thumbnail_path("TUM_Uterus_0001.svs", vcfg) == jpg


def test_resolve_thumbnail_falls_back_to_cache_root(tmp_path: Path):
    cache_root = tmp_path / "cache"
    slide_id = "TUM_Uterus_0002.svs"
    slide_dir = cache_root / slide_id
    slide_dir.mkdir(parents=True)
    png = slide_dir / "thumbnail.png"
    png.write_bytes(b"fake")

    vcfg = {"thumbnail": {"dataset_root": "", "variant": "thumbnails"}}
    assert resolve_thumbnail_path(cache_root, slide_id, vcfg=vcfg) == png


def test_build_slide_cache_uses_dataset_when_present(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "thumbnails").mkdir(parents=True)
    jpg = root / "thumbnails" / "CASE.jpg"
    jpg.write_bytes(b"fake")

    cache_root = tmp_path / "cache"
    vcfg = {"thumbnail": {"dataset_root": str(root), "variant": "thumbnails"}}
    sc = build_slide_cache(cache_root, "CASE.svs", vcfg=vcfg)
    assert sc.thumbnail_path == jpg
