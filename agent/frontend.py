"""Service helpers used by the Streamlit agent frontend."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from agent.backends import AnswerBackend, ZeroShotQwenBackend
from agent.controller import chain_to_dict, traverse
from agent.memory import CaseMemory
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
from vision.backends import VisualBundle


@dataclass(frozen=True)
class SlideOption:
    slide_id: str
    thumbnail_path: Path | None
    has_cache: bool
    has_existing_chain: bool


def normalize_openai_base_url(base_url: str) -> str:
    """Return an OpenAI-compatible API root ending in /v1."""
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("VLM endpoint is empty")
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        normalized_path = path
    elif path:
        normalized_path = f"{path}/v1"
    else:
        normalized_path = "/v1"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, normalized_path, parsed.query, parsed.fragment)
    )


def safe_upload_name(filename: str) -> str:
    stem = Path(filename or "uploaded_image.png").stem
    suffix = Path(filename or "uploaded_image.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        suffix = ".png"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{safe_stem or 'uploaded_image'}{suffix}"


def save_uploaded_image(data: bytes, filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / safe_upload_name(filename)
    path.write_bytes(data)
    return path


def run_fixed_image_chain(
    image_path: Path,
    *,
    backend: AnswerBackend,
    image_id: str | None = None,
    embedding_context: str = "",
    patch_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Run the full graph with one uploaded image attached to every VLM question."""
    visual = VisualBundle(
        thumbnail_path=image_path,
        patch_paths=list(patch_paths or []),
        metadata={"visual": "uploaded_image", "embedding_context": embedding_context},
    )
    steps = traverse(
        backend,
        case_memory=CaseMemory(),
        retriever_method="none",
        visual_method="none",
        fixed_visual_bundle=visual,
        skip_report_nodes=False,
    )
    return chain_to_dict(
        steps,
        slide_id=image_id or image_path.name,
        include_report=True,
    )


def run_remote_image_chain(
    image_path: Path,
    *,
    base_url: str,
    model_name: str,
    api_key: str = "EMPTY",
    embedding_context: str = "",
    patch_paths: list[Path] | None = None,
) -> dict[str, Any]:
    import openai

    client = openai.OpenAI(
        base_url=normalize_openai_base_url(base_url),
        api_key=api_key,
    )
    backend = ZeroShotQwenBackend(
        client,
        model_name,
        use_guided_choice=False,
        request_logprobs=False,
    )
    return run_fixed_image_chain(
        image_path,
        backend=backend,
        image_id=image_path.name,
        embedding_context=embedding_context,
        patch_paths=patch_paths,
    )


def save_baseline_result(
    chain: dict[str, Any],
    *,
    output_root: Path,
    image_name: str,
) -> Path:
    run_dir = output_root / Path(image_name).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "cot_chain.json"
    path.write_text(json.dumps(chain, indent=2) + "\n")
    return path


def load_uni2_summary(cache_root: Path, slide_id: str) -> dict[str, Any] | None:
    from vision.cache import slide_cache_dir

    path = slide_cache_dir(cache_root, slide_id) / "uni2_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_uni2_embedding_context(
    summary: dict[str, Any],
    *,
    max_dims_per_level: int = 24,
) -> str:
    """Load UNI2 slide embeddings and format a compact text context for a VLM."""
    import numpy as np
    import torch

    lines = [
        "UNI2 WSI embedding context is attached as numeric text features.",
        "Treat these as auxiliary global slide features; use the image evidence for visual verification.",
        f"Slide ID: {summary.get('slide_id', '')}",
    ]
    for item in summary.get("levels", []):
        level = str(item.get("level", ""))
        path = Path(str(item.get("slide_embedding_path", "")))
        n_patches = int(item.get("n_patches", 0) or 0)
        expected_dim = int(item.get("embedding_dim", 0) or 0)
        if not path.exists():
            lines.append(
                f"- {level}: embedding file missing; n_patches={n_patches}; dim={expected_dim}"
            )
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        values = np.asarray(tensor, dtype=np.float32).reshape(-1)
        if values.size == 0:
            lines.append(f"- {level}: empty embedding; n_patches={n_patches}")
            continue
        dim = int(values.size)
        prefix = ", ".join(f"{float(v):.4f}" for v in values[:max_dims_per_level])
        lines.append(
            f"- {level}: n_patches={n_patches}; dim={dim}; "
            f"mean={float(values.mean()):.4f}; std={float(values.std()):.4f}; "
            f"min={float(values.min()):.4f}; max={float(values.max()):.4f}; "
            f"l2={float(np.linalg.norm(values)):.4f}; "
            f"first_{min(max_dims_per_level, dim)}=[{prefix}]"
        )
    return "\n".join(lines)


def run_uni2_embedding(
    *,
    svs_path: Path,
    cache_root: Path,
    repo_path: Path,
    levels: list[str],
    max_patches: int,
    save_patch_images: bool,
) -> dict[str, Any]:
    from scripts.vision.encode_uni2_wsi import encode_slide_with_uni2
    from vision.encoders.uni2 import UNI2Encoder

    encoder = UNI2Encoder(repo_path=repo_path)
    return encode_slide_with_uni2(
        svs_path=svs_path,
        cache_root=cache_root,
        encoder=encoder,
        levels=levels,
        max_patches=max_patches,
        save_patch_images=save_patch_images,
    )


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
