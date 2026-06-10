from scripts.vision._common import default_encode_levels


def test_default_encode_levels_four_zooms():
    levels = default_encode_levels()
    assert levels == ["5x", "10x", "20x"]


def test_run_offline_wsi_requires_slide_or_index():
    import sys

    import pytest

    from scripts.preprocess.run_offline_wsi import main

    with pytest.raises(SystemExit):
        sys.argv = ["run_offline_wsi"]
        main()
