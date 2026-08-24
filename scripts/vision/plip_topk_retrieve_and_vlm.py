"""PLIP top-k patch retrieval for graph questions, with optional Qwen VLM verification."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from graph.loader import load_graph
from graph.schema import Node
from vision.wsi_io import (
    iter_tissue_patches,
    load_patch_at_coord,
    resolve_wsi_files,
    slide_id_from_path,
)


@dataclass
class RetrievedPatch:
    rank: int
    patch_index: int
    score: float
    coord: tuple[int, int]
    patch_size_lv0: int
    image_path: str


def _safe(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("×", "x")
    )


def _query_from_node(node: Node) -> str:
    parts = [node.question.strip()]
    if node.description.strip():
        parts.append(node.description.strip())
    if node.options:
        parts.append("Allowed diagnostic concepts: " + ", ".join(node.options))
    return " ".join(part for part in parts if part)


def _normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def write_clean_thumbnail(svs_path: Path, out_path: Path, *, max_edge_px: int = 1024) -> None:
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    try:
        thumb = slide.get_thumbnail((max_edge_px, max_edge_px)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(out_path, format="JPEG", quality=90, optimize=True)
    finally:
        slide.close()


class PlipEncoder:
    def __init__(self, model_path: Path, *, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(str(model_path), local_files_only=True).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)

    def encode_images(self, images: list[Image.Image], *, batch_size: int) -> np.ndarray:
        feats: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model.get_image_features(**inputs)
            feats.append(out.detach().float().cpu().numpy())
        return _normalize(np.vstack(feats).astype(np.float32))

    def encode_texts(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        feats: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model.get_text_features(**inputs)
            feats.append(out.detach().float().cpu().numpy())
        return _normalize(np.vstack(feats).astype(np.float32))


def _image_content(path: Path, *, max_edge: int = 1024) -> dict[str, Any]:
    """Return a compact JPEG data URL, stripping WSI thumbnail metadata."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge))
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90, optimize=True)
    raw = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{raw}"}}


def ask_qwen(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    thumbnail: Path,
    patches: list[Path],
    node: Node,
    retrieved: list[RetrievedPatch],
) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=endpoint.rstrip("/"), api_key=api_key)
    evidence = "\n".join(
        f"- rank {item.rank}: score={item.score:.4f}, coord={item.coord}, patch_size_lv0={item.patch_size_lv0}"
        for item in retrieved
    )
    prompt = (
        "You are a pathology visual question-answering assistant. "
        "Use the whole-slide thumbnail and the top PLIP-retrieved patches.\n\n"
        f"Question: {node.question}\n"
        f"Diagnostic guidance: {node.description}\n"
        f"Allowed answers: {', '.join(node.options) if node.options else 'free text'}\n"
        f"PLIP retrieved patch metadata:\n{evidence}\n\n"
        "Answer concisely. If allowed answers are listed, return one allowed answer key."
    )
    content = [_image_content(thumbnail)] + [_image_content(path) for path in patches]
    content.append({"type": "text", "text": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=96,
    )
    return (resp.choices[0].message.content or "").strip()


def process_slide(
    *,
    encoder: PlipEncoder,
    svs_path: Path,
    out_root: Path,
    nodes: list[Node],
    level: str,
    patch_size: int,
    batch_size: int,
    max_patches: int,
    top_k: int,
    background_threshold: int,
    ask_vlm_limit: int,
    endpoint: str,
    model: str,
    api_key: str,
    save_all_patches: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    slide_id = slide_id_from_path(svs_path)
    out_dir = out_root / slide_id / _safe(level)
    summary_path = out_dir / "plip_topk_summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text())

    patch_dir = out_dir / "patches"
    top_dir = out_dir / "topk"
    if save_all_patches:
        patch_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)
    thumbnail = out_dir / "thumbnail.jpg"
    write_clean_thumbnail(svs_path, thumbnail, max_edge_px=1024)

    image_emb_chunks: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    patch_size_lv0 = 0
    batch_images: list[Image.Image] = []

    def flush_batch() -> None:
        if not batch_images:
            return
        image_emb_chunks.append(encoder.encode_images(batch_images, batch_size=batch_size))
        batch_images.clear()

    for index, (img, coord, ps_lv0) in enumerate(
        iter_tissue_patches(
            svs_path,
            objective=level,
            patch_size=patch_size,
            stride=patch_size,
            background_threshold=background_threshold,
            max_patches=max_patches,
        )
    ):
        if save_all_patches:
            path = patch_dir / f"patch_{index:06d}.jpg"
            img.save(path, quality=90)
        batch_images.append(img)
        coords.append(coord)
        patch_size_lv0 = ps_lv0
        if len(batch_images) >= batch_size:
            flush_batch()
    flush_batch()

    if not coords:
        summary = {
            "slide_id": slide_id,
            "svs_path": str(svs_path),
            "level": level,
            "status": "no_tissue_patches",
            "warning": f"No tissue patches found at {level} with background_threshold={background_threshold}.",
            "patch_size": patch_size,
            "patch_size_lv0": 0,
            "n_patches_encoded": 0,
            "embedding_dim": 0,
            "thumbnail_path": str(thumbnail),
            "patch_embeddings_path": "",
            "nodes": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    image_emb = _normalize(np.vstack(image_emb_chunks).astype(np.float32))
    torch.save(
        {"embeddings": image_emb, "coords": np.asarray(coords, dtype=np.int64)},
        out_dir / "plip_patch_embeddings.pt",
    )

    node_queries = [_query_from_node(node) for node in nodes]
    text_emb = encoder.encode_texts(node_queries, batch_size=batch_size)
    sims = text_emb @ image_emb.T

    node_results: list[dict[str, Any]] = []
    for node_idx, node in enumerate(nodes):
        order = np.argsort(-sims[node_idx])[:top_k]
        retrieved: list[RetrievedPatch] = []
        node_top_dir = top_dir / node.id
        node_top_dir.mkdir(parents=True, exist_ok=True)
        for rank, patch_index in enumerate(order, start=1):
            image_path = node_top_dir / f"rank{rank:02d}_patch{int(patch_index):06d}.jpg"
            patch_img = load_patch_at_coord(
                svs_path,
                coords[int(patch_index)],
                objective=level,
                patch_size=patch_size,
            )
            patch_img.save(image_path, quality=95)
            retrieved.append(
                RetrievedPatch(
                    rank=rank,
                    patch_index=int(patch_index),
                    score=float(sims[node_idx, patch_index]),
                    coord=coords[int(patch_index)],
                    patch_size_lv0=int(patch_size_lv0),
                    image_path=str(image_path),
                )
            )
        vlm_answer = ""
        if ask_vlm_limit and node_idx < ask_vlm_limit:
            vlm_answer = ask_qwen(
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                thumbnail=thumbnail,
                patches=[Path(item.image_path) for item in retrieved],
                node=node,
                retrieved=retrieved,
            )
        node_results.append(
            {
                "node_id": node.id,
                "question": node.question,
                "description": node.description,
                "query": node_queries[node_idx],
                "top_patches": [asdict(item) for item in retrieved],
                "vlm_answer": vlm_answer,
            }
        )

    summary = {
        "slide_id": slide_id,
        "svs_path": str(svs_path),
        "level": level,
        "patch_size": patch_size,
        "patch_size_lv0": int(patch_size_lv0),
        "n_patches_encoded": len(coords),
        "embedding_dim": int(image_emb.shape[1]),
        "thumbnail_path": str(thumbnail),
        "patch_embeddings_path": str(out_dir / "plip_patch_embeddings.pt"),
        "nodes": node_results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plip-model", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--slide", default="")
    parser.add_argument(
        "--level",
        choices=["1x", "1.25x", "2.5x", "5x", "10x"],
        default="10x",
    )
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--background-threshold", type=int, default=220)
    parser.add_argument("--node-limit", type=int, default=0, help="0 means all graph nodes")
    parser.add_argument("--ask-vlm-limit", type=int, default=0, help="ask Qwen for first N nodes")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="/mnt/research/ljs/vlm_malignancy/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--save-all-patches", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    graph, _ = load_graph()
    nodes = list(graph.values())
    if args.node_limit:
        nodes = nodes[: args.node_limit]
    slides = resolve_wsi_files(args.data_dir, slide=args.slide, limit=args.limit)
    if not slides:
        raise SystemExit(f"No WSI files found under {args.data_dir}")

    encoder = PlipEncoder(args.plip_model)
    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for slide in slides:
        summary = process_slide(
            encoder=encoder,
            svs_path=slide,
            out_root=args.out_root,
            nodes=nodes,
            level=args.level,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            max_patches=args.max_patches,
            top_k=args.top_k,
            background_threshold=args.background_threshold,
            ask_vlm_limit=args.ask_vlm_limit,
            endpoint=args.endpoint,
            model=args.model,
            api_key=args.api_key,
            save_all_patches=args.save_all_patches,
            skip_existing=args.skip_existing,
        )
        print(json.dumps({
            "slide_id": summary["slide_id"],
            "level": summary["level"],
            "status": summary.get("status", "ok"),
            "n_patches_encoded": summary["n_patches_encoded"],
            "embedding_dim": summary["embedding_dim"],
            "summary_path": str(args.out_root / summary["slide_id"] / _safe(args.level) / "plip_topk_summary.json"),
            "first_node": summary["nodes"][0] if summary["nodes"] else None,
        }, indent=2))
        summaries.append(summary)
    (args.out_root / "run_summary.json").write_text(
        json.dumps({"n_slides": len(summaries), "slides": [s["slide_id"] for s in summaries]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
