"""Shared agent traversal setup for baselines and inference CLIs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.backends import (
    AnswerBackend,
    DummyBackend,
    FineTunedBackend,
    ZeroShotQwenBackend,
)
from agent.controller import chain_to_dict, traverse
from agent.memory import CaseMemory
from agent.types import Step
from vision.cache import SlideCache, build_slide_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_paths_config() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def load_vision_cache_root() -> Path | None:
    vision_cfg_path = REPO_ROOT / "configs" / "vision.yaml"
    if not vision_cfg_path.exists():
        return None
    with vision_cfg_path.open() as f:
        vcfg = yaml.safe_load(f)
    cr = vcfg.get("cache_root", "")
    return Path(cr).expanduser() if cr else None


def build_backend(name: str, cfg: dict | None = None) -> AnswerBackend:
    cfg = cfg or load_paths_config()
    if name == "dummy":
        return DummyBackend()
    if name == "qwen":
        import openai

        q = cfg["qwen"]
        api_base_url = os.environ.get("QWEN_API_BASE", q["api_base_url"])
        client = openai.OpenAI(base_url=api_base_url, api_key=q["api_key"])
        return ZeroShotQwenBackend(client, q["model_name"])
    if name == "lora":
        lora_cfg = cfg.get("lora", {}) if isinstance(cfg, dict) else {}
        adapter_dir = os.environ.get("LORA_ADAPTER_DIR") or lora_cfg.get("adapter_dir")
        if not adapter_dir:
            raise ValueError(
                "backend='lora' requires the adapter path via $LORA_ADAPTER_DIR "
                "or cfg['lora']['adapter_dir']"
            )
        base_model = (
            os.environ.get("LORA_BASE_MODEL")
            or lora_cfg.get("base_model")
            or cfg["qwen"]["model_path"]
        )
        return FineTunedBackend(adapter_dir, base_model=base_model)
    raise ValueError(f"Unknown backend: {name!r}")


@dataclass
class AgentRunResult:
    steps: list[Step]
    chain: dict[str, Any]
    retrieval_log: list[dict[str, Any]]


def run_agent_traversal(
    *,
    backend: str = "dummy",
    memory: str = "flat",
    memory_k: int = 5,
    visual: str = "thumbnail",
    retriever: str = "none",
    navigator: str = "graph_guided",
    slide_id: str = "",
    wsi_slide_id: str | None = None,
    cache_root: Path | None = None,
    wsi_data_dir: Path | None = None,
    skip_report_nodes: bool = False,
    search_all_patches: bool = False,
) -> AgentRunResult:
    cfg = load_paths_config()
    wsi_data_dir = wsi_data_dir or Path(cfg["cluster"]["data_dir"])
    cache_root = cache_root or load_vision_cache_root()
    visual_slide_id = wsi_slide_id or slide_id

    answer_backend = build_backend(backend, cfg)
    mem = CaseMemory.from_config(memory, memory_k=memory_k)
    slide_cache: SlideCache | None = (
        build_slide_cache(cache_root, visual_slide_id)
        if visual_slide_id and cache_root
        else None
    )

    retrieval_log: list[dict[str, Any]] = []
    steps = traverse(
        answer_backend,
        case_memory=mem,
        slide_cache=slide_cache,
        visual_method=visual,
        retriever_method=retriever,
        navigator_method=navigator,
        cache_root=cache_root,
        wsi_data_dir=wsi_data_dir,
        skip_report_nodes=skip_report_nodes,
        search_all_patches=search_all_patches,
        retrieval_log=retrieval_log,
    )
    chain = chain_to_dict(
        steps,
        slide_id=slide_id,
        include_report=not skip_report_nodes,
    )
    if wsi_slide_id and wsi_slide_id != slide_id:
        chain["inference_wsi"] = wsi_slide_id
    return AgentRunResult(steps=steps, chain=chain, retrieval_log=retrieval_log)


def default_runs_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_paths_config()
    work = Path(cfg["user"]["work_dir"])
    return work / "runs"


def write_run_outputs(
    result: AgentRunResult,
    runs_dir: Path,
    slide_id: str,
) -> Path:
    """Write cot_chain.json (+ optional retrieval_log) for a full agent run."""
    out_dir = runs_dir / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_path = out_dir / "cot_chain.json"
    chain_path.write_text(json.dumps(result.chain, indent=2) + "\n")
    if result.retrieval_log:
        (out_dir / "retrieval_log.json").write_text(
            json.dumps(result.retrieval_log, indent=2) + "\n"
        )
    return chain_path


# Backward-compatible alias
write_phase1_outputs = write_run_outputs
