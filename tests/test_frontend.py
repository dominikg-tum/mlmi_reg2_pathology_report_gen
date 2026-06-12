from pathlib import Path

from agent.frontend import (
    evidence_images,
    load_retrieval_log,
    load_saved_run,
    resolve_shared_path,
    run_fixed_image_chain,
    safe_upload_name,
    save_baseline_result,
    save_uploaded_image,
)
from agent.backends import DummyBackend


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
