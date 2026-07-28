import numpy as np
import pytest

from agent.report_writer import (
    SlideProjector,
    build_merged_report_prompt,
    build_report_prompt,
    mean_slide_embedding,
    merge_case_chains,
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
    assert "## Slide a.svs" in prompt


def test_build_merged_report_prompt_multi_slide():
    chains = [
        {"chain-of-thought": [{"question": "Q1", "answer": "A1"}]},
        {"chain-of-thought": [{"question": "Q2", "answer": "A2"}]},
    ]
    prompt = build_merged_report_prompt(
        chains, slide_ids=["s1.svs", "s2.svs"]
    )
    assert "## Slide s1.svs" in prompt
    assert "## Slide s2.svs" in prompt
    assert "Q1" in prompt and "A1" in prompt
    assert "Q2" in prompt and "A2" in prompt
    assert "2 whole-slide" in prompt


def test_merge_case_chains_prefixes_questions():
    chains = [
        {
            "chain-of-thought": [
                {"node_id": "n1", "question": "Organ?", "answer": "uterus"},
            ]
        },
        {
            "chain-of-thought": [
                {"node_id": "n2", "question": "Grade?", "answer": "1"},
            ]
        },
    ]
    merged = merge_case_chains(chains, ["a.svs", "b.svs"], "a.svs,b.svs")
    assert merged["slide_id"] == "a.svs,b.svs"
    assert merged["physical_slides"] == ["a.svs", "b.svs"]
    assert merged["fusion"] == "ss_llm"
    assert merged["chain-of-thought"][0]["question"].startswith("[a.svs]")
    assert merged["chain-of-thought"][0]["slide_id"] == "a.svs"
    assert merged["node_path"] == ["n1", "n2"]


def test_mean_slide_embedding():
    a = np.ones(4, dtype=np.float32)
    b = np.full(4, 3.0, dtype=np.float32)
    mean = mean_slide_embedding([a, None, b])
    assert mean is not None
    np.testing.assert_allclose(mean, np.full(4, 2.0))
    assert mean_slide_embedding([None, None]) is None
