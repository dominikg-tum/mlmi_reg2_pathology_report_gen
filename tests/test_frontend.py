from pathlib import Path

import pytest

from agent.frontend import (
    build_uni2_embedding_context,
    evidence_images,
    load_retrieval_log,
    load_saved_run,
    load_uni2_summary,
    normalize_openai_base_url,
    resolve_shared_path,
    run_fixed_image_chain,
    safe_upload_name,
    save_baseline_result,
    save_uploaded_image,
)
from agent.backends import DummyBackend
from vision.cache import slide_cache_dir


def test_load_saved_run_and_retrieval_log(tmp_path: Path):
    slide_dir = tmp_path / "CASE.svs"
    slide_dir.mkdir()
    (slide_dir / "cot_chain.json").write_text(
        '{"slide_id": "CASE.svs", "node_path": ["organ_procedure"]}'
    )
    (slide_dir / "retrieval_log.json").write_text(
        '[{"node_id": "compartment", "patches": []}]'
    )

    chain = load_saved_run(tmp_path, "CASE.svs")
    log = load_retrieval_log(tmp_path, "CASE.svs")

    assert chain["slide_id"] == "CASE.svs"
    assert log[0]["node_id"] == "compartment"


def test_evidence_images_flattens_existing_paths(tmp_path: Path):
    patch = tmp_path / "patch.png"
    patch.write_bytes(b"image")
    missing = tmp_path / "missing.png"
    log = [
        {
            "node_id": "compartment",
            "zoom_level": "10x",
            "patches": [
                {
                    "patch_path": str(patch),
                    "parent_patch_path": str(missing),
                    "similarity": 0.9,
                }
            ],
        }
    ]

    rows = evidence_images(log)

    assert len(rows) == 1
    assert rows[0]["path"] == patch
    assert rows[0]["scale"] == "selected"


def test_resolve_shared_path_preserves_existing_path(tmp_path: Path):
    assert resolve_shared_path(tmp_path) == tmp_path


def test_safe_upload_name():
    assert safe_upload_name("../../case image.JPG") == "case_image.jpg"
    assert safe_upload_name("case.exe") == "case.png"


def test_normalize_openai_base_url_adds_v1():
    assert normalize_openai_base_url("http://cluster:8000") == "http://cluster:8000/v1"
    assert normalize_openai_base_url("cluster:8000/v1") == "http://cluster:8000/v1"
    assert (
        normalize_openai_base_url("https://example.org/proxy/")
        == "https://example.org/proxy/v1"
    )


def test_load_uni2_summary_reads_slide_cache(tmp_path: Path):
    slide_dir = slide_cache_dir(tmp_path, "CASE.svs")
    slide_dir.mkdir()
    (slide_dir / "uni2_summary.json").write_text(
        '{"slide_id": "CASE.svs", "levels": [{"level": "10x", "n_patches": 2}]}'
    )

    summary = load_uni2_summary(tmp_path, "CASE.svs")

    assert summary is not None
    assert summary["levels"][0]["level"] == "10x"
    assert load_uni2_summary(tmp_path, "MISSING.svs") is None


def test_build_uni2_embedding_context_loads_slide_vector(tmp_path: Path):
    torch = pytest.importorskip("torch")

    embedding_path = tmp_path / "uni2_slide_embedding_10x.pt"
    torch.save([1.0, 2.0, 3.0, 4.0], embedding_path)
    summary = {
        "slide_id": "CASE.svs",
        "levels": [
            {
                "level": "10x",
                "n_patches": 3,
                "embedding_dim": 4,
                "slide_embedding_path": str(embedding_path),
            }
        ],
    }

    context = build_uni2_embedding_context(summary, max_dims_per_level=2)

    assert "UNI2 WSI embedding context" in context
    assert "10x: n_patches=3; dim=4" in context
    assert "first_2=[1.0000, 2.0000]" in context


def test_fixed_image_baseline_reaches_report_without_encoder(tmp_path: Path):
    image = save_uploaded_image(b"image", "case image.png", tmp_path / "uploads")
    chain = run_fixed_image_chain(image, backend=DummyBackend())

    assert chain["node_path"][-1] == "report"
    assert chain["report"] == "Sample pathology report."
    assert chain["slide_id"] == image.name

    output = save_baseline_result(
        chain,
        output_root=tmp_path / "runs",
        image_name=image.name,
    )
    assert output.exists()
    assert '"report": "Sample pathology report."' in output.read_text()
