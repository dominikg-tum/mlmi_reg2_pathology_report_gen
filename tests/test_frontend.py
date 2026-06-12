from pathlib import Path

from agent.frontend import (
    evidence_images,
    load_retrieval_log,
    load_saved_run,
    resolve_shared_path,
)


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
