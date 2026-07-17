"""Shared agent traversal setup for baselines and inference CLIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.backends import AnswerBackend, DummyBackend, ZeroShotQwenBackend
from agent.controller import chain_to_dict, traverse
from agent.memory import CaseMemory
from agent.types import Step
from vision.cache import SlideCache, build_slide_cache
from vision.thumbnail import _resolve_wsi_path

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
        client = openai.OpenAI(base_url=q["api_base_url"], api_key=q["api_key"])
        return ZeroShotQwenBackend(client, q["model_name"])
    raise ValueError(f"Unknown backend: {name!r}")


@dataclass
class AgentRunResult:
    steps: list[Step]
    chain: dict[str, Any]
    retrieval_log: list[dict[str, Any]]


def resolve_search_all_patches(
    *,
    kmeans_pool: bool = False,
    search_all_patches: bool = False,
) -> bool | None:
    """CLI override for retrieval pool. None → configs/vision.yaml default."""
    if kmeans_pool:
        return False
    if search_all_patches:
        return True
    return None


def run_agent_traversal(
    *,
    backend: str = "dummy",
    memory: str = "flat",
    visual: str = "thumbnail",
    retriever: str = "none",
    navigator: str = "graph_guided",
    slide_id: str = "",
    cache_root: Path | None = None,
    wsi_data_dir: Path | None = None,
    skip_report_nodes: bool = False,
    search_all_patches: bool | None = None,
    node_react: bool = False,
    structured_answer: bool = False,
    paired_regions: bool = False,
) -> AgentRunResult:
    cfg = load_paths_config()
    wsi_data_dir = wsi_data_dir or Path(cfg["cluster"]["data_dir"])
    cache_root = cache_root or load_vision_cache_root()

    answer_backend = build_backend(backend, cfg)
    mem = CaseMemory.from_config(memory)
    slide_cache: SlideCache | None = (
        build_slide_cache(cache_root, slide_id) if slide_id and cache_root else None
    )

    # Resolve on-disk WSI so node_react zoom / return_images can attach patch PNGs.
    wsi_path = _resolve_wsi_path(
        slide_cache, wsi_path=None, wsi_data_dir=wsi_data_dir
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
        wsi_path=wsi_path,
        wsi_data_dir=wsi_data_dir,
        skip_report_nodes=skip_report_nodes,
        search_all_patches=search_all_patches,
        retrieval_log=retrieval_log,
        node_react=node_react,
        structured_answer=structured_answer,
        paired_regions=paired_regions,
    )
    chain = chain_to_dict(
        steps,
        slide_id=slide_id,
        include_report=not skip_report_nodes,
    )
    return AgentRunResult(steps=steps, chain=chain, retrieval_log=retrieval_log)


def default_runs_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_paths_config()
    work = Path(cfg["user"]["work_dir"])
    return work / "runs"


def write_phase1_outputs(
    result: AgentRunResult,
    runs_dir: Path,
    slide_id: str,
) -> Path:
    out_dir = runs_dir / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_path = out_dir / "cot_chain.json"
    chain_path.write_text(json.dumps(result.chain, indent=2) + "\n")
    report = str(result.chain.get("report", "") or "").strip()
    if report:
        (out_dir / "report.txt").write_text(report + "\n")
    if result.retrieval_log:
        (out_dir / "retrieval_log.json").write_text(
            json.dumps(result.retrieval_log, indent=2) + "\n"
        )
    return chain_path
