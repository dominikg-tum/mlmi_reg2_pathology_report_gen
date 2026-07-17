"""Tests for UUID ↔ TUM_Uterus name map."""

from __future__ import annotations

from pathlib import Path

from vision import wsi_mapping as wm
from vision.wsi_io import resolve_wsi_files, slide_id_from_path


def _write_map(tmp_path: Path) -> Path:
    csv_path = tmp_path / "wsi_name_map.csv"
    csv_path.write_text(
        "wsi_index,tum_num,slide_id,disk_name,specimen_slide_id,case_key,block_id,"
        "img_id,tum_image_id,disease_label,report_duplicate\n"
        "0,0001,TUM_Uterus_0001.svs,aaaa-bbbb.svs,TUM_Uterus_0001_p0001_T1-A-1.svs,"
        "p0001,T1-A-1,ID1,1,malignant_tumor,0\n"
        "1,0002,TUM_Uterus_0002.svs,cccc-dddd.svs,TUM_Uterus_0002_p0002_T1-A-1.svs,"
        "p0002,T1-A-1,ID2,2,benign_tumor,0\n"
    )
    return csv_path


def test_canonical_and_disk_lookup(tmp_path: Path) -> None:
    csv_path = str(_write_map(tmp_path))
    wm.load_wsi_name_map.cache_clear()
    assert wm.canonical_slide_id("aaaa-bbbb.svs", csv_path=csv_path) == "TUM_Uterus_0001.svs"
    assert wm.disk_filename("TUM_Uterus_0002.svs", csv_path=csv_path) == "cccc-dddd.svs"
    assert wm.mapped_slide_ids(csv_path=csv_path) == [
        "TUM_Uterus_0001.svs",
        "TUM_Uterus_0002.svs",
    ]


def test_resolve_uses_map_index_and_disk_name(tmp_path: Path, monkeypatch) -> None:
    csv_path = str(_write_map(tmp_path))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    disk = data_dir / "aaaa-bbbb.svs"
    disk.write_bytes(b"x")
    (data_dir / "cccc-dddd.svs").write_bytes(b"y")

    wm.load_wsi_name_map.cache_clear()
    monkeypatch.setattr(wm, "DEFAULT_NAME_MAP_CSV", Path(csv_path))

    files = resolve_wsi_files(data_dir, wsi_index=0)
    assert files == [disk]
    assert slide_id_from_path(disk) == "TUM_Uterus_0001.svs"


def test_repo_name_map_has_464_rows() -> None:
    wm.load_wsi_name_map.cache_clear()
    rows = wm.load_wsi_name_map()
    assert len(rows) == 464
    assert rows[0].slide_id == "TUM_Uterus_0001.svs"
    assert rows[0].disk_name.endswith(".svs")
    assert rows[-1].slide_id == "TUM_Uterus_0464.svs"
