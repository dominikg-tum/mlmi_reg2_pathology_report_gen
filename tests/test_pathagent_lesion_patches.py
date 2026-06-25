from pathlib import Path

import numpy as np

from scripts.vision import extract_pathagent_lesion_patches as lesion_export


class _FakeImage:
    def __init__(self, label: str):
        self.label = label

    def save(self, path: Path) -> None:
        path.write_text(self.label)


class _FakeScorer:
    def score(self, query: str, patch_images: list, *, batch_size: int = 16) -> np.ndarray:
        assert "lesion" in query
        return np.asarray([0.1, 0.9, 0.4], dtype=np.float32)


def test_extract_lesion_patches_5x_writes_top_patch(monkeypatch, tmp_path: Path):
    svs = tmp_path / "CASE.svs"
    svs.write_text("not a real slide")

    def fake_iter(*_args, **kwargs):
        assert kwargs["objective"] == "5x"
        yield _FakeImage("a"), (0, 0), 1792
        yield _FakeImage("b"), (1792, 0), 1792
        yield _FakeImage("c"), (3584, 0), 1792

    monkeypatch.setattr(lesion_export, "iter_tissue_patches", fake_iter)

    manifest = lesion_export.extract_lesion_patches_5x(
        svs_path=svs,
        cache_root=tmp_path / "cache",
        scorer=_FakeScorer(),
        query="lesion",
        top_k=1,
    )

    selected = manifest["selected"][0]
    assert selected["coord"] == [1792, 0]
    patch_path = Path(selected["patch_path"])
    assert patch_path.exists()
    assert patch_path.read_text() == "b"
    assert (tmp_path / "cache" / "CASE.svs" / "lesion_patches_5x.json").exists()
