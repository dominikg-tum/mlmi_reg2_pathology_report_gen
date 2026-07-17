"""Inline-tile path must use coords-only tiling + shared encode selection."""

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


@pytest.fixture(autouse=True)
def _torch_ready():
    _ensure_torch()


def test_inline_tile_uses_coords_and_shared_encode_flow(tmp_path: Path) -> None:
    from scripts.vision.encode_patches_offline import _encode_one

    out_dir = tmp_path / "slide"
    svs = tmp_path / "CASE.svs"
    svs.write_bytes(b"")
    full_coords = [(0, 0), (10, 10), (20, 20), (30, 30)]
    selected = [(0, 0), (20, 20)]
    enc = MagicMock()
    enc.encode_patches.side_effect = lambda patches, batch_size=16: np.zeros(
        (len(patches), 4), dtype=np.float32
    )

    with (
        patch(
            "scripts.vision.encode_patches_offline.extract_patch_coords",
            return_value=(full_coords, 512),
        ) as extract_coords,
        patch(
            "scripts.vision.encode_patches_offline.coords_for_encode",
            return_value=(
                selected,
                "stratified",
                {
                    "n_patches_tiled": len(full_coords),
                    "n_patches_encoded": len(selected),
                    "sampling_mode": "stratified",
                    "full_encode_threshold": 1024,
                    "max_patches_per_slide": 4096,
                    "grid_cells": [8, 8],
                },
            ),
        ) as select,
        patch(
            "scripts.vision.encode_patches_offline.load_patches_from_coords",
            return_value=[MagicMock(), MagicMock()],
        ) as load_chunks,
    ):
        _encode_one(
            enc,
            svs,
            out_dir,
            "20x",
            batch_size=2,
            max_patches=2,
        )

    extract_coords.assert_called_once()
    select.assert_called_once()
    assert load_chunks.call_count >= 1
    from scripts.vision import encode_patches_offline as mod

    assert not hasattr(mod, "extract_patches")
    assert (out_dir / "coords_20x.pt").exists()
    assert (out_dir / "patch_embeddings_20x.pt").exists()
    meta = json.loads((out_dir / "meta_20x.json").read_text())
    assert meta["sampling_mode"] == "stratified"
    assert meta["n_patches_tiled"] == 4
    assert meta["n_patches_encoded"] == 2
    assert meta["patch_size_lv0"] == 512
