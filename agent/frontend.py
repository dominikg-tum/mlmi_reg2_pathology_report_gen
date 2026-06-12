"""Service helpers used by the Streamlit agent frontend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agent.report_writer import MedGemmaReportBackend, write_report
from baselines.agent_runner import (
    AgentRunResult,
    default_runs_dir,
    load_paths_config,
    load_vision_cache_root,
    run_agent_traversal,
    write_phase1_outputs,
)
from vision.cache import dataset_thumbnail_dir, slide_id_to_stem


@dataclass(frozen=True)
class SlideOption:
    slide_id: str
    thumbnail_path: Path | None
    has_cache: bool
    has_existing_chain: bool


def resolve_shared_path(path: Path) -> Path:
    """Resolve cluster /mnt/projects paths through the macOS SMB mount when needed."""
    path = path.expanduser()
    if path.exists():
        return path
    cluster_root = Path("/mnt/projects")
    try:
        relative = path.relative_to(cluster_root)
    except ValueError:
        return path
    mounted = Path("/Volumes/projects") / relative
    return mounted if mounted.exists() else path


def _slide_id_from_thumbnail(path: Path) -> str:
    return f"{path.stem}.svs"


def _iter_wsi_ids(data_dir: Path) -> Iterable[str]:
    if not data_dir.is_dir():
        return []
    direct = list(data_dir.glob("*.svs"))
    nested = list(data_dir.glob("*/*.svs"))
    return (path.name for path in direct + nested)


def discover_slides(
    *,
    cache_root: Path | None = None,
    runs_dir: Path | None = None,
    data_dir: Path | None = None,
) -> list[SlideOption]:
    """Discover slide IDs from thumbnail banks, caches, runs, and WSI storage."""
    cfg = load_paths_config()
    configured_cache = cache_root or load_vision_cache_root()
    cache_root = resolve_shared_path(configured_cache) if configured_cache else None
    runs_dir = resolve_shared_path(runs_dir or default_runs_dir(cfg))
    data_dir = resolve_shared_path(data_dir or Path(cfg["cluster"]["data_dir"]))

    slide_ids: set[str] = set()
    thumbnail_by_id: dict[str, Path] = {}
    cache_ids: set[str] = set()
    run_ids: set[str] = set()

    bank = dataset_thumbnail_dir()
    if bank is not None:
        bank = resolve_shared_path(bank)
    if bank and bank.is_dir():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for path in bank.glob(pattern):
                slide_id = _slide_id_from_thumbnail(path)
                slide_ids.add(slide_id)
                thumbnail_by_id[slide_id] = path

    if cache_root and cache_root.is_dir():
        for path in cache_root.iterdir():
            if path.is_dir():
                cache_ids.add(path.name)
        slide_ids.update(cache_ids)

    if runs_dir.is_dir():
        for path in runs_dir.iterdir():
            if path.is_dir():
                run_ids.add(path.name)
        slide_ids.update(run_ids)

    # Thumbnail banks are the fast, canonical UI index. Only inspect WSI storage
    # when no lighter-weight source is available; recursive SMB scans stall reruns.
    if not slide_ids:
        slide_ids.update(_iter_wsi_ids(data_dir))

    options: list[SlideOption] = []
    for slide_id in sorted(slide_ids):
        thumbnail = thumbnail_by_id.get(slide_id)
        if thumbnail is None and cache_root and slide_id in cache_ids:
            slide_cache_dir = cache_root / slide_id
            for name in ("thumbnail.png", "thumbnail.jpg", "thumbnail.jpeg"):
                candidate = slide_cache_dir / name
                if candidate.exists():
                    thumbnail = candidate
                    break
        options.append(
            SlideOption(
                slide_id=slide_id,
                thumbnail_path=thumbnail,
                has_cache=slide_id in cache_ids,
                has_existing_chain=(
                    slide_id in run_ids
                    and (runs_dir / slide_id / "cot_chain.json").exists()
                ),
            )
        )
    return options


def run_phase1(
    slide_id: str,
    *,
    backend: str,
    memory: str,
    visual: str,
    retriever: str,
    navigator: str = "graph_guided",
    cache_root: Path | None = None,
    runs_dir: Path | None = None,
    wsi_data_dir: Path | None = None,
    search_all_patches: bool = False,
) -> tuple[AgentRunResult, Path]:
    result = run_agent_traversal(
        backend=backend,
        memory=memory,
        visual=visual,
        retriever=retriever,
        navigator=navigator,
        slide_id=slide_id,
        cache_root=cache_root,
        wsi_data_dir=wsi_data_dir,
        skip_report_nodes=True,
        search_all_patches=search_all_patches,
    )
    output_root = resolve_shared_path(runs_dir or default_runs_dir())
    chain_path = write_phase1_outputs(result, output_root, slide_id)
    return result, chain_path


def load_saved_run(runs_dir: Path, slide_id: str) -> dict[str, Any] | None:
    path = runs_dir / slide_id / "cot_chain.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_retrieval_log(runs_dir: Path, slide_id: str) -> list[dict[str, Any]]:
    path = runs_dir / slide_id / "retrieval_log.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else []


def evidence_images(retrieval_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten saved retrieval metadata into image rows for display."""
    rows: list[dict[str, Any]] = []
    for event in retrieval_log:
        for patch in event.get("patches") or []:
            for key, scale in (
                ("patch_path", "selected"),
                ("parent_patch_path", "parent"),
                ("grandparent_patch_path", "grandparent"),
            ):
                raw_path = patch.get(key)
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.exists():
                    continue
                rows.append(
                    {
                        "path": path,
                        "node_id": event.get("node_id", ""),
                        "zoom_level": event.get("zoom_level", ""),
                        "scale": scale,
                        "similarity": patch.get("similarity"),
                    }
                )
    return rows


def generate_phase2_report(
    chain: dict[str, Any],
    *,
    slide_id: str,
    runs_dir: Path,
    model_path: str,
    max_new_tokens: int = 1024,
) -> tuple[str, Path]:
    backend = MedGemmaReportBackend(model_path)
    report = backend.generate_report(chain, max_new_tokens=max_new_tokens)
    report_path = runs_dir / slide_id / "report.txt"
    write_report(report_path, report)
    return report, report_path


def load_saved_report(runs_dir: Path, slide_id: str) -> str:
    path = runs_dir / slide_id / "report.txt"
    return path.read_text().strip() if path.exists() else ""


def slide_label(option: SlideOption) -> str:
    status = []
    if option.has_cache:
        status.append("cache")
    if option.has_existing_chain:
        status.append("chain")
    suffix = f"  [{' / '.join(status)}]" if status else ""
    return f"{slide_id_to_stem(option.slide_id)}{suffix}"
