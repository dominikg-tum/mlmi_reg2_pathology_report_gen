"""Build LoRA training JSONL from WP3 chains using the SAME visual pathway as inference.

One ``ChainSample`` per (train slide, non-report node on its GT path). The supervision
label is the GT ``answer`` already stored in ``chains.jsonl`` (extracted from the real
report) — we do NOT need patch-level labels. The patch/thumbnail images are selected by
the exact inference-time retriever (``graph_guided`` at ``node.zoom_level``) so that the
fine-tuned model trains on the same evidence it will see at test time.

Heavy dependencies (TITAN text encoder, openslide) are imported lazily, so this module
imports cleanly on a laptop; real data generation runs on the cluster inside enroot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.controller import build_query
from agent.types import Step
from graph import GRAPH, Node
from graph.schema import NodeKind
from vision.cache import SlideCache, build_slide_cache, slide_id_to_stem

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical multi-slide → primary-WSI routing lives in data/case_slides.py, but that
# module is gitignored (data/*) and absent from fresh clones. Import it when present
# (shared cluster repo) for exact train/serve parity; otherwise fall back to the
# self-contained reimplementation below, which matches tests/test_case_slides.py.
try:  # pragma: no cover - depends on untracked file being present
    from data.case_slides import primary_wsi_for_baseline as _PRIMARY_WSI_FN
except Exception:  # ModuleNotFoundError in pinned clones
    _PRIMARY_WSI_FN = None


@dataclass
class ChainSample:
    slide_id: str
    node_id: str
    question: str
    target_answer: str
    visual_paths: list[str]
    episodic_context: str


# --------------------------------------------------------------------------- #
# chains.jsonl reading
# --------------------------------------------------------------------------- #
def _iter_chain_records(chains_path: Path):
    with chains_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _episodic_context(prior_steps: list[Step]) -> str:
    """Match ZeroShotQwenBackend's prior-answer block (Q/A history)."""
    return "\n".join(f"Q: {s.question}\nA: {s.answer}" for s in prior_steps)


# --------------------------------------------------------------------------- #
# multi-slide case -> primary vision WSI (parity with inference baseline)
# --------------------------------------------------------------------------- #
def _default_wsi_map_path() -> Path:
    return REPO_ROOT / "data" / "manifests" / "wsi_id_map.json"


def _load_wsi_map(path: Path | None) -> dict[str, str]:
    """Load the UUID(.svs) -> TUM_Uterus_XXXX.svs mapping; {} if file missing."""
    path = Path(path) if path else _default_wsi_map_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("mapping", payload) if isinstance(payload, dict) else {}
    return {str(k): str(v) for k, v in mapping.items()}


def _resolve_vision_wsi(
    case_slide_id: str, wsi_map: dict[str, str], *, primary_index: int
) -> str:
    """Map a (possibly multi-slide) chains case id to the single WSI used for vision.

    chains.jsonl stores the *case* id: a comma-separated list of UUID ``.svs`` names.
    For Phase A baseline parity we pick one primary slide (default index 1 = corpus for
    fractional curettage; clamped for shorter lists) and translate its UUID to the
    ``TUM_Uterus_XXXX.svs`` name used by the offline cache / thumbnail bank.
    """
    s = str(case_slide_id)
    if _PRIMARY_WSI_FN is not None:
        return _PRIMARY_WSI_FN(s, index=primary_index)
    ids = [x.strip() for x in s.split(",") if x.strip()]
    if not ids:
        return ""
    chosen = ids[min(primary_index, len(ids) - 1)]
    return wsi_map.get(chosen, chosen)


# --------------------------------------------------------------------------- #
# image materialization (parity with vision/vlm_messages._image_part)
# --------------------------------------------------------------------------- #
def _export_image(src: Path, dst: Path, *, max_edge_px: int) -> bool:
    """Resize/recompress one image to the inference budget and save under dst. Returns ok."""
    from PIL import Image

    from vision.vlm_messages import _load_rgb_image

    try:
        img = _load_rgb_image(src)
    except (FileNotFoundError, OSError):
        return False
    w, h = img.size
    if max(w, h) > max_edge_px:
        scale = max_edge_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=85)
    return True


def _materialize_visuals(
    visual_bundle,
    *,
    image_root: Path,
    slide_id: str,
    node_id: str,
    max_edge_px: int,
) -> list[str]:
    """Copy the inference-selected images into a stable per-(slide,node) folder."""
    from agent.backends import _visual_image_paths

    selected = _visual_image_paths(visual_bundle)
    out_dir = image_root / slide_id_to_stem(slide_id) / node_id
    saved: list[str] = []
    for i, src in enumerate(selected):
        dst = out_dir / f"img_{i}{src.suffix.lower() if src.suffix else '.jpg'}"
        dst = dst.with_suffix(".jpg")
        if _export_image(Path(src), dst, max_edge_px=max_edge_px):
            saved.append(str(dst))
    return saved


# --------------------------------------------------------------------------- #
# retriever / visual provider construction (lazy heavy imports)
# --------------------------------------------------------------------------- #
def _build_retriever(retriever_method: str, *, search_all_patches: bool):
    from retrieval.base import get_retriever

    kwargs: dict[str, Any] = {}
    if retriever_method in ("titan_cosine", "graph_guided"):
        from vision.encoders.titan import TitanEncoder

        encoder = TitanEncoder()
        kwargs["text_encoder"] = encoder.encode_text
        kwargs["search_all_patches"] = search_all_patches
    return get_retriever(retriever_method, **kwargs)


def _build_visual_provider(visual_method: str, *, cache_root, wsi_data_dir):
    from vision.backends import get_visual_provider

    return get_visual_provider(
        visual_method, cache_root, wsi_data_dir=wsi_data_dir
    )


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def build_training_jsonl(
    chains_path: Path,
    output_path: Path,
    *,
    split: str = "train",
    visual_method: str = "patch_retrieve",
    retriever_method: str = "graph_guided",
    cache_root: Path | None = None,
    wsi_data_dir: Path | None = None,
    image_root: Path | None = None,
    search_all_patches: bool = False,
    include_report_node: bool = False,
    max_image_edge_px: int = 512,
    limit: int = 0,
    primary_index: int = 1,
    wsi_id_map: dict[str, str] | None = None,
    wsi_id_map_path: Path | None = None,
    # Dependency injection (tests / custom pipelines)
    visual_provider: Any = None,
    retriever: Any = None,
) -> int:
    """Unroll WP3 chains into per-node LoRA samples; return number of samples written.

    Only ``split`` slides with ``extraction_status == 'ok'`` are used. Each non-report
    node on the GT path becomes one ``ChainSample`` whose ``visual_paths`` come from the
    inference retriever, and whose ``target_answer`` is the GT edge key.
    """
    chains_path = Path(chains_path)
    output_path = Path(output_path)
    image_root = Path(image_root) if image_root else output_path.parent / "images"

    if wsi_id_map is None:
        wsi_id_map = _load_wsi_map(wsi_id_map_path)

    if retriever is None:
        retriever = _build_retriever(
            retriever_method, search_all_patches=search_all_patches
        )
    if visual_provider is None:
        visual_provider = _build_visual_provider(
            visual_method, cache_root=cache_root, wsi_data_dir=wsi_data_dir
        )

    n_written = 0
    n_slides = 0
    n_skipped_no_visual = 0
    n_unmapped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for record in _iter_chain_records(chains_path):
            if record.get("split") != split:
                continue
            if record.get("extraction_status", "ok") != "ok":
                continue
            slide_id = record.get("slide_id", "")
            cot = record.get("chain-of-thought", [])
            if not slide_id or not cot:
                continue
            if limit and n_slides >= limit:
                break
            n_slides += 1

            # chains slide_id is the *case* id (comma-separated UUID .svs). Route vision
            # to the primary single WSI (TUM_Uterus_XXXX.svs) the offline cache uses.
            vision_wsi = _resolve_vision_wsi(
                slide_id, wsi_id_map, primary_index=primary_index
            )
            if not vision_wsi:
                n_unmapped += 1

            slide_cache: SlideCache | None = (
                build_slide_cache(cache_root, vision_wsi)
                if (cache_root and vision_wsi)
                else None
            )

            prior_steps: list[Step] = []
            for step in cot:
                node_id = step.get("node_id", "")
                answer = (step.get("answer") or "").strip()
                node: Node | None = GRAPH.get(node_id)
                if node is None or not answer:
                    continue

                is_report = node.node_kind == NodeKind.REPORT
                if not is_report:
                    query = build_query(node, prior_steps)
                    visual_bundle = visual_provider.for_node(
                        node, slide_cache, query=query, retriever=retriever
                    )
                    visual_paths = _materialize_visuals(
                        visual_bundle,
                        image_root=image_root,
                        slide_id=vision_wsi or slide_id,
                        node_id=node_id,
                        max_edge_px=max_image_edge_px,
                    )
                    if node.requires_visual_evidence and not visual_paths:
                        n_skipped_no_visual += 1
                    else:
                        sample = ChainSample(
                            slide_id=slide_id,
                            node_id=node_id,
                            question=node.question,
                            target_answer=answer,
                            visual_paths=visual_paths,
                            episodic_context=_episodic_context(prior_steps),
                        )
                        out_f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                        n_written += 1

                # Keep history growing for context/parity, including report node.
                if include_report_node or not is_report:
                    prior_steps.append(
                        Step(node_id, node.question, answer, 1.0)
                    )

    print(
        f"Wrote {n_written} samples from {n_slides} {split} slides "
        f"-> {output_path} (images under {image_root}); "
        f"skipped {n_skipped_no_visual} nodes with no visual evidence; "
        f"{n_unmapped} cases had no WSI-id-map entry."
    )
    return n_written


def load_chain_samples(path: Path) -> list[ChainSample]:
    samples: list[ChainSample] = []
    for record in _iter_chain_records(Path(path)):
        samples.append(
            ChainSample(
                slide_id=record["slide_id"],
                node_id=record["node_id"],
                question=record["question"],
                target_answer=record["target_answer"],
                visual_paths=list(record.get("visual_paths", [])),
                episodic_context=record.get("episodic_context", ""),
            )
        )
    return samples


# --------------------------------------------------------------------------- #
# CLI (runs on the cluster inside enroot)
# --------------------------------------------------------------------------- #
def _default_chains_path() -> Path:
    return REPO_ROOT / "data" / "labels" / "chains.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LoRA training JSONL from WP3 chains (train split only)."
    )
    parser.add_argument("--chains", type=Path, default=_default_chains_path())
    parser.add_argument("--output", type=Path, required=True, help="Output samples.jsonl path")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--visual", default="patch_retrieve")
    parser.add_argument("--retriever", default="graph_guided")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--wsi-data-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--search-all-patches", action="store_true")
    parser.add_argument("--include-report-node", action="store_true")
    parser.add_argument("--max-image-edge-px", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Process only N slides (0=all)")
    parser.add_argument(
        "--primary-index",
        type=int,
        default=1,
        help="Which slide of a multi-slide case to use for vision (1=corpus, baseline default)",
    )
    parser.add_argument(
        "--wsi-id-map",
        type=Path,
        default=None,
        help="UUID->TUM_Uterus map JSON (default: data/manifests/wsi_id_map.json)",
    )
    args = parser.parse_args()

    cache_root = args.cache_root
    wsi_data_dir = args.wsi_data_dir
    if cache_root is None or wsi_data_dir is None:
        from baselines.agent_runner import load_paths_config, load_vision_cache_root

        cfg = load_paths_config()
        if cache_root is None:
            cache_root = load_vision_cache_root()
        if wsi_data_dir is None:
            wsi_data_dir = Path(cfg["cluster"]["data_dir"])

    build_training_jsonl(
        args.chains,
        args.output,
        split=args.split,
        visual_method=args.visual,
        retriever_method=args.retriever,
        cache_root=cache_root,
        wsi_data_dir=wsi_data_dir,
        image_root=args.image_root,
        search_all_patches=args.search_all_patches,
        include_report_node=args.include_report_node,
        max_image_edge_px=args.max_image_edge_px,
        limit=args.limit,
        primary_index=args.primary_index,
        wsi_id_map_path=args.wsi_id_map,
    )


if __name__ == "__main__":
    main()
