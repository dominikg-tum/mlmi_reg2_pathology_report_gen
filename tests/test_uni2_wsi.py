from pathlib import Path

import numpy as np

from scripts.vision import encode_uni2_wsi
from vision.encoders.uni2 import _resolve_assets_dir, mean_pool_embeddings
from vision.wsi_io import objective_downsample


class _DummySlide:
    properties = {"openslide.mpp-x": "0.25"}


class _FakeImage:
    def __init__(self, label: str):
        self.label = label

    def save(self, path: Path) -> None:
        path.write_bytes(self.label.encode("utf-8"))


class _FakeEncoder:
    def encode_patches(self, patch_images: list, *, batch_size: int = 16) -> np.ndarray:
        rows = []
        for i, _patch in enumerate(patch_images):
            rows.append([float(i), float(i + 1), float(batch_size)])
        return np.asarray(rows, dtype=np.float32)


def test_low_magnification_downsamples():
    slide = _DummySlide()
    assert objective_downsample(slide, "1.25x") == 32.0
    assert objective_downsample(slide, "2.5x") == 16.0
    assert objective_downsample(slide, "5x") == 8.0
    assert objective_downsample(slide, "10x") == 4.0


def test_mean_pool_embeddings():
    emb = np.asarray([[1, 2, 3], [3, 4, 5]], dtype=np.float32)
    np.testing.assert_allclose(mean_pool_embeddings(emb), np.asarray([2, 3, 4], dtype=np.float32))


def test_resolve_uni2_assets_dir(tmp_path: Path):
    weights_dir = tmp_path / "uni2-h"
    weights_dir.mkdir()
    ckpt = weights_dir / "pytorch_model.bin"
    ckpt.write_bytes(b"weights")

    assert _resolve_assets_dir(weights_dir, "uni2-h") == tmp_path
    assert _resolve_assets_dir(ckpt, "uni2-h") == tmp_path
    assert _resolve_assets_dir(tmp_path, "uni2-h") == tmp_path

    upper_parent = tmp_path / "upper"
    upper_parent.mkdir()
    upper_dir = upper_parent / "UNI2-h"
    upper_dir.mkdir()
    (upper_dir / "pytorch_model.bin").write_bytes(b"weights")
    assets_dir = _resolve_assets_dir(upper_dir, "uni2-h")
    assert (assets_dir / "uni2-h" / "pytorch_model.bin").exists()


def test_encode_slide_with_uni2_writes_artifacts(monkeypatch, tmp_path: Path):
    svs = tmp_path / "case.svs"
    svs.write_text("not a real slide")

    def fake_iter(*_args, **_kwargs):
        yield _FakeImage("a"), (0, 0), 1792
        yield _FakeImage("b"), (1792, 0), 1792

    def fake_thumb(_svs_path, out_path, *, max_edge_px=1024):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"thumb")

    monkeypatch.setattr(encode_uni2_wsi, "iter_tissue_patches", fake_iter)
    monkeypatch.setattr(encode_uni2_wsi, "write_thumbnail", fake_thumb)

    summary = encode_uni2_wsi.encode_slide_with_uni2(
        svs_path=svs,
        cache_root=tmp_path / "cache",
        encoder=_FakeEncoder(),
        levels=["1.25x", "2.5x"],
        patch_size=224,
        batch_size=4,
        max_patches=0,
        save_patch_images=True,
    )

    out_dir = tmp_path / "cache" / "case.svs"
    assert (out_dir / "thumbnail.png").exists()
    assert (out_dir / "uni2_patch_embeddings_1p25x.pt").exists()
    assert (out_dir / "uni2_coords_2p5x.pt").exists()
    assert (out_dir / "uni2_slide_embedding_1p25x.pt").exists()
    assert (out_dir / "patches" / "1p25x" / "patch_000000.png").exists()
    assert summary["levels"][0]["n_patches"] == 2
    assert summary["levels"][0]["embedding_dim"] == 3
