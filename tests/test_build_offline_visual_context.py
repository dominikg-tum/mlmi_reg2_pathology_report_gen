from pathlib import Path

from scripts.vision import build_offline_visual_context as visual_context


def test_build_visual_context_for_slide_combines_lesion_and_uni2(monkeypatch, tmp_path: Path):
    svs = tmp_path / "CASE.svs"
    svs.write_text("not a real slide")

    def fake_lesion(**kwargs):
        assert kwargs["svs_path"] == svs
        assert kwargs["query"] == "lesion"
        return {"selected": [{"patch_path": "lesion.png"}]}

    def fake_uni2(**kwargs):
        assert kwargs["svs_path"] == svs
        assert kwargs["levels"] == ["5x"]
        return {"levels": [{"level": "5x"}]}

    monkeypatch.setattr(visual_context, "extract_lesion_patches_5x", fake_lesion)
    monkeypatch.setattr(visual_context, "encode_slide_with_uni2", fake_uni2)

    result = visual_context.build_visual_context_for_slide(
        svs_path=svs,
        cache_root=tmp_path / "cache",
        lesion_scorer=object(),
        uni2_encoder=object(),
        lesion_query="lesion",
        levels=["5x"],
    )

    assert result["slide_id"] == "CASE.svs"
    assert result["lesion_patches"]["selected"][0]["patch_path"] == "lesion.png"
    assert result["uni2"]["levels"][0]["level"] == "5x"
