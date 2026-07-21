"""Build a multimodal SFT dataset from WP3 chains, using the SAME visual pathway
as Phase 1 inference (train/serve parity).

For every train slide in ``chains.jsonl`` and every node on its ground-truth path,
we recreate the exact input the Phase-1 node answerer sees under
``--structured-answer`` / ``--node-react``:

    system = prompts.STEP_A_SYSTEM
    user   = prompts.format_step_a_user(node, prior_steps)
    images = thumbnail (+ CONCH top-k retrieved patches for patch_retrieve nodes)

and set the assistant target to the ground-truth answer (JSON ``{answer_key, ...}``
by default, matching ``agent/backends.complete_json``). One ``ChainSample`` per
(slide, node) — no cross-slide pooling (see training/README.md v1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import prompts
from graph import load_graph
from graph.schema import InteractionType, Node
from vision.mag_config import fixed_retrieval_pool

# Nodes whose answer the Phase-1 loop resolves via a single allowed key. These are
# the nodes served through ``complete_json`` (structured answer / node ReAct), so
# they are the ones a LoRA node-answerer should be trained on.
_CHOICE_INTERACTIONS = (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN)


@dataclass
class ChainSample:
    """One (slide, node) training example, ready to serialize to JSONL."""

    slide_id: str
    node_id: str
    question: str
    target_answer: str
    visual_paths: list[str]
    episodic_context: str
    system: str = ""
    user: str = ""
    target: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "node_id": self.node_id,
            "question": self.question,
            "target_answer": self.target_answer,
            "images": list(self.visual_paths),
            "episodic_context": self.episodic_context,
            "system": self.system,
            "user": self.user,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, raw: dict[str, Any]) -> "ChainSample":
        return cls(
            slide_id=raw.get("slide_id", ""),
            node_id=raw.get("node_id", ""),
            question=raw.get("question", ""),
            target_answer=raw.get("target_answer", ""),
            visual_paths=list(raw.get("images") or raw.get("visual_paths") or []),
            episodic_context=raw.get("episodic_context", ""),
            system=raw.get("system", ""),
            user=raw.get("user", ""),
            target=raw.get("target", ""),
            metadata=dict(raw.get("metadata") or {}),
        )


def render_target(answer: str, *, answer_format: str = "json") -> str:
    """Assistant-side supervision string.

    ``json``  → ``{"answer_key": <gt>, "rationale": "", "confidence": 1.0}`` so the
    fine-tuned model emits the same structure ``complete_json`` parses at inference.
    ``key``   → the bare answer key (parity with plain ``--backend qwen``).
    """
    if answer_format == "json":
        return json.dumps(
            {"answer_key": answer, "rationale": "", "confidence": 1.0},
            ensure_ascii=False,
        )
    if answer_format == "key":
        return answer
    raise ValueError(f"Unknown answer_format {answer_format!r}; use 'json' or 'key'")


def build_chat_messages(
    system: str,
    user: str,
    target: str | None,
    n_images: int,
) -> list[dict[str, Any]]:
    """OpenAI/Qwen-style chat messages with ``n_images`` image placeholders.

    Pure helper (no I/O) shared by the trainer collator and the inference backend,
    so training and serving build identical prompts. When ``target`` is ``None`` the
    assistant turn is omitted (inference: caller adds the generation prompt).
    """
    user_content: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    user_content.append({"type": "text", "text": user})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]
    if target is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        )
    return messages


def _read_chain_records(chains_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with chains_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _save_patch_images(retrieved: list, out_dir: Path) -> list[str]:
    """Persist retrieved patch (+ optional parent/grandparent) crops to ``out_dir``.

    Mirrors ``vision.thumbnail._bundle_from_retrieved`` ordering, but writes to an
    arbitrary directory so training image crops never land in a (possibly shared,
    read-only) embeddings cache.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, rp in enumerate(retrieved):
        if getattr(rp, "patch_image", None) is not None:
            p = out_dir / f"patch_{i}_{rp.level}.png"
            rp.patch_image.save(p)
            paths.append(str(p))
        if getattr(rp, "parent_image", None) is not None:
            pl = rp.parent_level or "parent"
            pp = out_dir / f"patch_{i}_parent_{pl}.png"
            rp.parent_image.save(pp)
            paths.append(str(pp))
        if getattr(rp, "grandparent_image", None) is not None:
            gl = rp.grandparent_level or "grandparent"
            gp = out_dir / f"patch_{i}_grandparent_{gl}.png"
            rp.grandparent_image.save(gp)
            paths.append(str(gp))
    return paths


def _node_image_paths(
    node: Node,
    slide_cache,
    *,
    retriever,
    wsi_path,
    images_out_dir: Path,
) -> list[str]:
    """Recreate Phase-1 visual evidence for one node → list of on-disk image paths.

    thumbnail_only nodes get the whole-slide thumbnail; patch_retrieve/both nodes
    additionally get CONCH top-k patches, written under ``images_out_dir`` (owned by
    the caller — NOT the embeddings cache dir).
    """
    images: list[str] = []
    thumb = getattr(slide_cache, "thumbnail_path", None) if slide_cache else None
    if thumb is not None:
        images.append(str(thumb))

    if node.needs_patch_retrieval() and retriever is not None and slide_cache is not None:
        retrieved = retriever.retrieve(
            node.retrieval_text,
            slide_cache,
            level=fixed_retrieval_pool(),
            wsi_path=wsi_path,
            return_images=wsi_path is not None,
            tier=node.tier.value,
            node_kind=node.node_kind.value,
        )
        images.extend(_save_patch_images(retrieved, images_out_dir))

    return images


def build_training_jsonl(
    chains_path: Path,
    output_path: Path,
    *,
    retriever=None,
    resolve_slide=None,
    images_out_root: Path | None = None,
    visual_method: str = "patch_retrieve",
    answer_format: str = "json",
    splits: tuple[str, ...] = ("train",),
    choice_nodes_only: bool = True,
    limit: int = 0,
    graph: dict[str, Node] | None = None,
) -> int:
    """Unroll WP3 chains into per-node multimodal SFT samples.

    Parameters
    ----------
    retriever
        A ``PatchRetriever`` (e.g. graph_guided over TITAN/CONCH). Required for
        ``patch_retrieve`` nodes; if ``None`` only thumbnails are attached.
    resolve_slide
        ``callable(slide_id) -> (SlideCache | None, wsi_path | None)``. Owns the
        UUID→canonical name mapping: the returned ``SlideCache`` must point at the
        canonical (``TUM_Uterus_XXXX``) cache dir, and ``wsi_path`` at the on-disk
        ``.svs``. If ``None``, samples are built text/thumbnail-only.
    images_out_root
        Directory the caller owns; per-node patch crops are written under
        ``images_out_root/<slide>/<node>/``. Defaults next to ``output_path`` so we
        never write into a (possibly shared, read-only) embeddings cache.

    Returns the number of samples written.
    """
    graph = graph or load_graph()[0]
    records = _read_chain_records(chains_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_out_root = Path(images_out_root) if images_out_root else (
        output_path.parent / "train_images"
    )
    n_written = 0
    n_slides = 0

    with output_path.open("w", encoding="utf-8") as out:
        for rec in records:
            if rec.get("extraction_status", "ok") != "ok":
                continue
            if splits and str(rec.get("split", "")) not in splits:
                continue
            if limit and n_slides >= limit:
                break

            slide_id = rec.get("slide_id", "")
            cot = rec.get("chain-of-thought") or []
            if not slide_id or not cot:
                continue

            slide_cache, wsi_path = (
                resolve_slide(slide_id) if resolve_slide else (None, None)
            )

            prior_steps: list[tuple[str, str]] = []
            n_slides += 1
            for item in cot:
                node_id = item.get("node_id", "")
                answer = item.get("answer", "")
                node = graph.get(node_id)
                if node is None or not answer:
                    if node_id and answer:
                        prior_steps.append((node_id, answer))
                    continue

                is_choice = node.interaction in _CHOICE_INTERACTIONS
                if choice_nodes_only and not is_choice:
                    prior_steps.append((node_id, answer))
                    continue

                safe_slide = slide_id.replace(",", "_").replace("/", "_")
                safe_node = node_id.replace("/", "_")
                images = _node_image_paths(
                    node,
                    slide_cache,
                    retriever=retriever if visual_method != "none" else None,
                    wsi_path=wsi_path,
                    images_out_dir=images_out_root / safe_slide / safe_node,
                )
                system = prompts.STEP_A_SYSTEM
                user = prompts.format_step_a_user(node=node, prior_steps=prior_steps)
                target = render_target(answer, answer_format=answer_format)
                episodic = "\n".join(f"{nid} -> {ans}" for nid, ans in prior_steps)

                sample = ChainSample(
                    slide_id=slide_id,
                    node_id=node_id,
                    question=node.question,
                    target_answer=answer,
                    visual_paths=images,
                    episodic_context=episodic,
                    system=system,
                    user=user,
                    target=target,
                    metadata={
                        "split": rec.get("split", ""),
                        "interaction": node.interaction.value,
                        "visual_policy": node.visual_policy.value,
                        "n_images": len(images),
                    },
                )
                out.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
                n_written += 1
                prior_steps.append((node_id, answer))

    return n_written


def load_chain_samples(path: Path) -> list[ChainSample]:
    return [ChainSample.from_record(r) for r in _read_chain_records(Path(path))]
