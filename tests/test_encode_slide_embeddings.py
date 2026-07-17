"""Validation before TITAN slide aggregation in encode_slide_embeddings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_HAS_TORCH = False
try:
    import torch as _torch

    _HAS_TORCH = True
except ImportError:
    _torch = None  # type: ignore[assignment]


def _ensure_torch():
    """Use real torch when installed; otherwise a load/save stub."""
    if _HAS_TORCH:
        return _torch
    if isinstance(sys.modules.get("torch"), MagicMock):
        return sys.modules["torch"]
    torch_mod = MagicMock()
    store: dict[str, object] = {}

    def _save(obj, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        store[str(p)] = obj
        p.write_bytes(b"stub")

    def _load(path, map_location=None, weights_only=False):
        return store[str(Path(path))]

    torch_mod.save.side_effect = _save
    torch_mod.load.side_effect = _load
    sys.modules["torch"] = torch_mod
    return torch_mod


def _save_pt(path: Path, obj) -> None:
    _ensure_torch().save(obj, path)


def _fake_encoder(dim: int = 4) -> MagicMock:
    enc = MagicMock()
    enc.model_id = "fake-titan"

    def _encode_patches(patches, batch_size=32):
        return np.zeros((len(patches), dim), dtype=np.float32)

    def _encode_slide(patch_emb, coords, patch_size_lv0):
        assert patch_size_lv0 > 0
        assert len(coords) == int(patch_emb.shape[0])
        return np.ones(dim, dtype=np.float32)

    enc.encode_patches.side_effect = _encode_patches
    enc.encode_slide.side_effect = _encode_slide
    return enc


@pytest.fixture(autouse=True)
def _torch_ready():
    _ensure_torch()


def test_legacy_cache_coord_mismatch_rebuilds(tmp_path: Path) -> None:
    """Raw emb tensor + full-pool coords_20x.pt must not aggregate mismatched rows."""
    from scripts.vision.encode_slide_embeddings import encode_one_slide

    out_dir = tmp_path / "slide"
    out_dir.mkdir()
    _save_pt(out_dir / "patch_embeddings_20x.pt", np.zeros((2, 4), dtype=np.float32))
    _save_pt(
        out_dir / "coords_20x.pt",
        np.asarray([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=np.int64),
    )
    (out_dir / "meta_20x.json").write_text(
        json.dumps({"patch_size_lv0": 512}) + "\n"
    )
    svs = tmp_path / "CASE.svs"
    svs.write_bytes(b"")

    patch_img = MagicMock()
    with (
        patch(
            "scripts.vision.encode_slide_embeddings.coords_for_encode",
            return_value=(
                [(0, 0), (10, 10)],
                "stratified",
                {"n_patches_tiled": 4},
            ),
        ),
        patch(
            "scripts.vision.encode_slide_embeddings.load_patches_from_coords",
            return_value=[patch_img, patch_img],
        ),
    ):
        encode_one_slide(
            _fake_encoder(),
            svs,
            out_dir,
            max_patches=2,
            batch_size=8,
            write_thumb=False,
            max_edge_px=64,
        )

    assert (out_dir / "slide_embedding.pt").exists()


def test_cache_mismatch_without_patch_size_raises(tmp_path: Path) -> None:
    from scripts.vision.encode_slide_embeddings import encode_one_slide

    out_dir = tmp_path / "slide"
    out_dir.mkdir()
    _save_pt(out_dir / "patch_embeddings_20x.pt", np.zeros((2, 4), dtype=np.float32))
    _save_pt(
        out_dir / "coords_20x.pt",
        np.asarray([[0, 0], [10, 10], [20, 20]], dtype=np.int64),
    )
    svs = tmp_path / "CASE.svs"
    svs.write_bytes(b"")

    with pytest.raises(RuntimeError, match="cannot rebuild without positive"):
        encode_one_slide(
            _fake_encoder(),
            svs,
            out_dir,
            max_patches=2,
            batch_size=8,
            write_thumb=False,
            max_edge_px=64,
        )


def test_coords_without_meta_patch_size_raises(tmp_path: Path) -> None:
    from scripts.vision.encode_slide_embeddings import encode_one_slide

    out_dir = tmp_path / "slide"
    out_dir.mkdir()
    _save_pt(
        out_dir / "coords_20x.pt",
        np.asarray([[0, 0], [10, 10]], dtype=np.int64),
    )
    svs = tmp_path / "CASE.svs"
    svs.write_bytes(b"")

    with (
        patch(
            "scripts.vision.encode_slide_embeddings.coords_for_encode",
            return_value=(
                [(0, 0), (10, 10)],
                "all",
                {"n_patches_tiled": 2},
            ),
        ),
        pytest.raises(RuntimeError, match="missing positive patch_size_lv0"),
    ):
        encode_one_slide(
            _fake_encoder(),
            svs,
            out_dir,
            max_patches=2,
            batch_size=8,
            write_thumb=False,
            max_edge_px=64,
        )


def test_canonical_cache_ok(tmp_path: Path) -> None:
    from scripts.vision.encode_slide_embeddings import encode_one_slide

    out_dir = tmp_path / "slide"
    out_dir.mkdir()
    _save_pt(
        out_dir / "patch_embeddings_20x.pt",
        {
            "embeddings": np.zeros((2, 4), dtype=np.float32),
            "coords": np.asarray([[0, 0], [10, 10]], dtype=np.int64),
        },
    )
    (out_dir / "meta_20x.json").write_text(
        json.dumps({"patch_size_lv0": 512}) + "\n"
    )
    svs = tmp_path / "CASE.svs"
    svs.write_bytes(b"")

    encode_one_slide(
        _fake_encoder(),
        svs,
        out_dir,
        max_patches=2,
        batch_size=8,
        write_thumb=False,
        max_edge_px=64,
    )
    assert (out_dir / "slide_embedding.pt").exists()
