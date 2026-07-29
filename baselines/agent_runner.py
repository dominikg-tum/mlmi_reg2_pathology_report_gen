"""Shared agent traversal setup for baselines and inference CLIs."""

from __future__ import annotations

import json
import logging
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
from agent.report_writer import write_case_chain
from agent.slide_selector import (
    build_selected_case_chain,
    select_slide_chain,
    selection_from_case_chain,
    write_case_meta,
)
from agent.types import Step
from extraction.case_ids import (
    CaseSpec,
    case_run_dir,
    case_spec_from_key,
    physical_run_dir,
)
from vision.cache import SlideCache, build_slide_cache
from vision.mag_config import fixed_retrieval_pool
from vision.thumbnail import _resolve_wsi_path

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def load_paths_config() -> dict:
    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        cfg = yaml.safe_load(f)
    # Cross-node vLLM: set by start_qwen_server.sh / run_baseline_batch.sh
    override = (os.environ.get("QWEN_API_BASE_URL") or "").strip()
    if override:
        cfg.setdefault("qwen", {})["api_base_url"] = override
    return cfg


def load_vision_cache_root() -> Path | None:
    vision_cfg_path = REPO_ROOT / "configs" / "vision.yaml"
    if not vision_cfg_path.exists():
        return None
    with vision_cfg_path.open() as f:
        vcfg = yaml.safe_load(f)
    cr = vcfg.get("cache_root", "")
    return Path(cr).expanduser() if cr else None


def resolve_adapter_dir(cfg: dict | None = None) -> str | None:
    """LoRA adapter path: env override beats paths.yaml finetuned.adapter_dir."""
    cfg = cfg or load_paths_config()
    for key in ("MLMI_ADAPTER_DIR", "LORA_ADAPTER_DIR"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return raw
    ft = cfg.get("finetuned", {}) or {}
    raw = str(ft.get("adapter_dir") or "").strip()
    return raw or None


def patch_embeddings_exist(
    cache_root: Path | None,
    slide_id: str,
    *,
    level: str | None = None,
) -> bool:
    """True when offline CONCH embeddings for the retrieval pool are on disk."""
    if cache_root is None or not slide_id:
        return False
    slide_cache = build_slide_cache(cache_root, slide_id)
    path = slide_cache.embedding_path_for_level(level or fixed_retrieval_pool())
    return path is not None and path.exists()


def build_backend(name: str, cfg: dict | None = None) -> AnswerBackend:
    cfg = cfg or load_paths_config()
    if name == "dummy":
        return DummyBackend()
    if name == "qwen":
        import openai

        q = cfg["qwen"]
        client = openai.OpenAI(base_url=q["api_base_url"], api_key=q["api_key"])
        return ZeroShotQwenBackend(client, q["model_name"])
    if name == "finetuned":
        ft = cfg.get("finetuned", {})
        base_model = ft.get("base_model") or cfg["models"]["qwen3_vl_8b"]
        adapter_dir = resolve_adapter_dir(cfg)
        if not adapter_dir:
            raise ValueError(
                "finetuned backend requires an adapter. Set MLMI_ADAPTER_DIR / "
                "LORA_ADAPTER_DIR or finetuned.adapter_dir in configs/paths.yaml."
            )
        adapter_path = Path(adapter_dir)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_path}. "
                "Train with scripts/cluster/train_lora.sh or override "
                "MLMI_ADAPTER_DIR / LORA_ADAPTER_DIR."
            )
        return FineTunedBackend(base_model, str(adapter_path))
    raise ValueError(f"Unknown backend: {name!r}")


def build_selector_backend(backend: str = "qwen") -> AnswerBackend | None:
    """SS-LLM selector model, shared by all ablations. None → deterministic fallback.

    LoRA adapters are skipped so a second large model is never loaded for selection.
    """
    name = "qwen" if backend == "finetuned" else backend
    try:
        return build_backend(name)
    except Exception as exc:
        logger.warning(
            "SS-LLM selector backend %r unavailable (%s); using deterministic fallback",
            name,
            exc,
        )
        return None


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
    case_key: str = "",
    cache_root: Path | None = None,
    wsi_data_dir: Path | None = None,
    skip_report_nodes: bool = False,
    search_all_patches: bool | None = None,
    node_react: bool = False,
    structured_answer: bool = False,
    paired_regions: bool = False,
    react_max_iters: int | None = None,
) -> AgentRunResult:
    cfg = load_paths_config()
    wsi_data_dir = wsi_data_dir or Path(cfg["cluster"]["data_dir"])
    cache_root = cache_root or load_vision_cache_root()

    answer_backend = build_backend(backend, cfg)
    exclude_key = (case_key or slide_id or "").strip() or None
    mem = CaseMemory.from_config(memory, exclude_case_key=exclude_key)
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
        react_max_iters=react_max_iters,
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
    report_path = out_dir / "report.txt"
    if report:
        report_path.write_text(report + "\n")
    else:
        report_path.unlink(missing_ok=True)
    if result.retrieval_log:
        (out_dir / "retrieval_log.json").write_text(
            json.dumps(result.retrieval_log, indent=2) + "\n"
        )
    return chain_path


def write_phase1_outputs_to_dir(
    result: AgentRunResult,
    out_dir: Path,
) -> Path:
    """Write Phase 1 artifacts into an explicit directory (case/slides layout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_path = out_dir / "cot_chain.json"
    chain_path.write_text(json.dumps(result.chain, indent=2) + "\n")
    report = str(result.chain.get("report", "") or "").strip()
    report_path = out_dir / "report.txt"
    if report:
        report_path.write_text(report + "\n")
    else:
        report_path.unlink(missing_ok=True)
    if result.retrieval_log:
        (out_dir / "retrieval_log.json").write_text(
            json.dumps(result.retrieval_log, indent=2) + "\n"
        )
    return chain_path


def run_case_phase1(
    case: CaseSpec | str,
    *,
    runs_dir: Path,
    backend: str = "qwen",
    memory: str = "flat",
    visual: str = "thumbnail",
    retriever: str = "none",
    navigator: str = "graph_guided",
    skip_report_nodes: bool = False,
    search_all_patches: bool | None = None,
    node_react: bool = False,
    structured_answer: bool = False,
    paired_regions: bool = False,
    react_max_iters: int | None = None,
    skip_existing: bool = False,
    cache_root: Path | None = None,
    wsi_data_dir: Path | None = None,
) -> Path:
    """SS-LLM Phase 1: full graph per WSI, then pick one case-level chain."""
    if isinstance(case, str):
        case = case_spec_from_key(case)

    case_dir = case_run_dir(runs_dir, case.case_key)
    case_chain_path = case_dir / "cot_chain.json"
    cache_root = cache_root or load_vision_cache_root()

    skipped_slides: list[dict[str, Any]] = []
    runnable_slides = list(case.physical_slides)
    if visual == "patch_retrieve":
        runnable_slides = []
        for physical_id in case.physical_slides:
            if patch_embeddings_exist(cache_root, physical_id):
                runnable_slides.append(physical_id)
            else:
                skipped_slides.append(
                    {
                        "slide_id": physical_id,
                        "skipped": True,
                        "reason": "no_patch_cache",
                    }
                )
        if not runnable_slides:
            missing = ", ".join(case.physical_slides)
            raise FileNotFoundError(
                f"No patch_embeddings cache for any slide in case {case.case_key!r} "
                f"({missing}). Run offline encode or skip this case."
            )

    def _all_runnable_chains_exist() -> bool:
        return all(
            (physical_run_dir(runs_dir, case.case_key, pid) / "cot_chain.json").exists()
            for pid in runnable_slides
        )

    # Old concatenated case chains must be migrated to SS-LLM Pick.
    if skip_existing and case_chain_path.exists() and _all_runnable_chains_exist():
        stored = selection_from_case_chain(
            json.loads(case_chain_path.read_text()), runnable_slides
        )
        if stored is not None:
            meta_path = case_dir / "case_meta.json"
            if not meta_path.exists():
                write_case_meta(
                    meta_path,
                    case_key=case.case_key,
                    physical_slides=case.physical_slides,
                    selection=stored,
                    skipped_slides=skipped_slides,
                )
            return case_chain_path

    chains: list[dict[str, Any]] = []
    chain_slide_ids: list[str] = []
    for physical_id in runnable_slides:
        phys_dir = physical_run_dir(runs_dir, case.case_key, physical_id)
        phys_chain = phys_dir / "cot_chain.json"
        if skip_existing and phys_chain.exists():
            chains.append(json.loads(phys_chain.read_text()))
            chain_slide_ids.append(physical_id)
            continue

        result = run_agent_traversal(
            backend=backend,
            memory=memory,
            visual=visual,
            retriever=retriever,
            navigator=navigator,
            slide_id=physical_id,
            case_key=case.case_key,
            cache_root=cache_root,
            wsi_data_dir=wsi_data_dir,
            skip_report_nodes=skip_report_nodes,
            search_all_patches=search_all_patches,
            node_react=node_react,
            structured_answer=structured_answer,
            paired_regions=paired_regions,
            react_max_iters=react_max_iters,
        )
        write_phase1_outputs_to_dir(result, phys_dir)
        chains.append(result.chain)
        chain_slide_ids.append(physical_id)

    selection = select_slide_chain(
        chains,
        chain_slide_ids,
        backend=build_selector_backend(backend),
    )
    selected_index = chain_slide_ids.index(selection.chosen_slide_id)
    case_chain = build_selected_case_chain(
        chains[selected_index],
        case_key=case.case_key,
        physical_slides=case.physical_slides,
        selection=selection,
    )
    write_case_chain(case_chain_path, case_chain)
    write_case_meta(
        case_dir / "case_meta.json",
        case_key=case.case_key,
        physical_slides=case.physical_slides,
        selection=selection,
        skipped_slides=skipped_slides,
    )
    return case_chain_path
