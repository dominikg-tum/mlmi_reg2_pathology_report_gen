import numpy as np
import pytest

from agent.report_writer import (
    SlideProjector,
    build_report_prompt,
)


def test_slide_projector_shape():
    pytest.importorskip("torch")
    proj = SlideProjector(in_dim=1024, out_dim=4096)
    out = proj.project(np.ones(1024, dtype=np.float32))
    assert out.shape == (4096,)


def test_build_report_prompt_includes_chain():
    chain = {
        "slide_id": "a.svs",
        "chain-of-thought": [
            {"question": "Organ?", "answer": "uterus"},
        ],
    }
    prompt = build_report_prompt(chain)
    assert "Organ?" in prompt
    assert "uterus" in prompt
    assert "CAP-format" in prompt
    assert "Do not define medical terms" in prompt
    assert "Selected physical slide: a.svs" in prompt
