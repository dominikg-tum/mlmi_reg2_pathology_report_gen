"""run_agent_traversal must resolve wsi_path for node_react zoom / patch images."""

from __future__ import annotations

from pathlib import Path

import baselines.agent_runner as agent_runner
from vision.cache import SlideCache


def test_run_agent_traversal_passes_resolved_wsi_path(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    wsi = data_dir / "case01.svs"
    wsi.write_bytes(b"fake-svs")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    captured: dict = {}

    def _fake_traverse(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(agent_runner, "traverse", _fake_traverse)
    monkeypatch.setattr(agent_runner, "build_backend", lambda *a, **k: object())
    monkeypatch.setattr(
        agent_runner.CaseMemory,
        "from_config",
        classmethod(lambda cls, name, **kwargs: object()),
    )
    monkeypatch.setattr(
        agent_runner,
        "build_slide_cache",
        lambda root, sid: SlideCache(slide_id=sid, cache_dir=cache_root / sid, thumbnail_path=None),
    )
    monkeypatch.setattr(agent_runner, "load_paths_config", lambda: {"cluster": {"data_dir": str(data_dir)}})
    monkeypatch.setattr(agent_runner, "chain_to_dict", lambda *a, **k: {})

    _ = agent_runner.run_agent_traversal(
        backend="dummy",
        slide_id="case01.svs",
        cache_root=cache_root,
        wsi_data_dir=data_dir,
        node_react=True,
    )

    assert captured.get("wsi_path") == wsi
    assert Path(captured["wsi_path"]).exists()
