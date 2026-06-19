"""WP3: extract Q->A / chain-of-thought from english_report (DOMI).

Produces supervised labels for eval and LoRA training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from extraction.graph_walk import steps_to_chain_dict, walk_graph
from graph import load_graph

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or REPO_ROOT / "configs" / "paths.yaml"
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    qwen = cfg.setdefault("qwen", {})
    if os.environ.get("QWEN_API_BASE_URL"):
        qwen["api_base_url"] = os.environ["QWEN_API_BASE_URL"]
    if os.environ.get("QWEN_MODEL_NAME"):
        qwen["model_name"] = os.environ["QWEN_MODEL_NAME"]
    if os.environ.get("QWEN_API_KEY"):
        qwen["api_key"] = os.environ["QWEN_API_KEY"]
    return cfg


def build_client(cfg: dict[str, Any]):
    import openai

    qwen = cfg["qwen"]
    return openai.OpenAI(base_url=qwen["api_base_url"], api_key=qwen["api_key"])


def main() -> None:
    """Smoke test: one sample report → graph walk → print JSON chain."""
    cfg = load_config()
    client = build_client(cfg)
    model = cfg["qwen"]["model_name"]
    sample_report = (
        "Uterus, hysterectomy. Endometrioid adenocarcinoma, FIGO grade 2, "
        "with deep myometrial invasion. Leiomyomata also present."
    )
    graph, root_id = load_graph()
    result = walk_graph(sample_report, graph, root_id, client, model)
    record = steps_to_chain_dict(
        "SMOKE.svs",
        result.steps,
        sample_report,
        "train",
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
