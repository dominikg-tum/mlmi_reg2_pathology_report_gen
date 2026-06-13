"""Offline vision script helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.vision._common import load_coords_from_pt


def test_load_coords_from_pt(tmp_path: Path) -> None:
    coords = np.array([[10, 20], [30, 40]], dtype=np.int64)
    path = tmp_path / "coords_20x.pt"
    torch.save(coords, path)
    assert load_coords_from_pt(path) == [(10, 20), (30, 40)]
