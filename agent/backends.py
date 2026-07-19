"""VLM backends: zero-shot Qwen, dummy, future LoRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from agent.types import Step

if TYPE_CHECKING:
    import openai
from graph.schema import InteractionType, Node, NodeKind
from vision.backends import VisualBundle
from vision.vlm_messages import build_user_content

SYSTEM_PROMPT = (
    "You are a pathology visual question-answering assistant. Analyze only the "
    "provided uterine whole-slide overview and retrieved tissue patches. Use prior "
    "answers as context, but do not invent findings that are not supported by the "
    "images. For a multiple-choice question, return exactly one allowed answer key "
    "and no explanation."
)


class AnswerBackend(Protocol):
    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]: ...


class ZeroShotQwenBackend:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]:
        prompt = _build_answer_prompt(node, visual, memory, extra_context)
        image_paths = _visual_image_paths(visual)
        content = build_user_content(prompt, image_paths)

        extra_body: dict[str, Any] = {}
        if node.interaction in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN):
            if node.options:
                extra_body["guided_choice"] = node.options

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            logprobs=True,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        confidence = _first_token_prob(choice)
        return answer, confidence

    def complete_json(
        self,
        node: Node,
        visual: VisualBundle | None,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any], float, str]:
        image_paths = _visual_image_paths(visual)
        content = build_user_content(user_prompt, image_paths)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            logprobs=True,
        )
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        confidence = _first_token_prob(choice)
        return _parse_json_soft(raw), confidence, raw


def _build_answer_prompt(
    node: Node,
    visual: VisualBundle | None,
    memory: list[Step],
    extra_context: str,
) -> str:
    """Shared user prompt for the plain (non-structured) answer path."""
    history = "\n".join(f"Q: {s.question}\nA: {s.answer}" for s in memory)
    visual_note = _visual_prompt_note(visual)
    prompt_parts = [f"Visual evidence:{visual_note or ' none attached.'}"]
    if history:
        prompt_parts.append(f"Prior diagnostic answers:\n{history}")
    if extra_context:
        prompt_parts.append(f"Additional context:\n{extra_context}")
    if node.description:
        prompt_parts.append(f"Diagnostic guidance:\n{node.description}")
    prompt_parts.append(f"Current question:\n{node.question}")
    if node.options:
        prompt_parts.append(
            "Allowed answer keys:\n" + "\n".join(f"- {option}" for option in node.options)
        )
    if node.node_kind == NodeKind.REPORT:
        prompt_parts.append(
            "Combine the visual findings and all prior diagnostic answers into a "
            "concise final pathology report. State the specimen/procedure when "
            "supported, followed by the principal diagnosis and key qualifiers. "
            "Do not mention the reasoning process or answer keys."
        )
    elif node.interaction == InteractionType.FREE_TEXT:
        prompt_parts.append("Return a concise pathology answer.")
    else:
        prompt_parts.append("Return exactly one allowed answer key.")
    return "\n\n".join(prompt_parts)


def _parse_json_soft(raw: str) -> dict[str, Any]:
    """Soft-fail JSON parse so traverse/node_react can retry instead of aborting."""
    parsed: dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except Exception:
                parsed = {}
        else:
            parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed


def _visual_prompt_note(visual: VisualBundle | None) -> str:
    if visual is None:
        return ""
    mode = visual.metadata.get("visual", "")
    if mode == "slide_embed" and visual.slide_embedding is not None:
        dim = visual.metadata.get("slide_embedding_dim", "?")
        n_ev = visual.metadata.get("evidence_patch_count", 0)
        return (
            f"\n[Visual: TITAN slide embedding (dim={dim}) is precomputed offline; "
            f"attached are overview thumbnail + {n_ev} tissue evidence patches.]"
        )
    if visual.thumbnail_path:
        patch_count = len(visual.patch_paths)
        suffix = f" and {patch_count} retrieved patch images" if patch_count else ""
        return f" whole-slide thumbnail{suffix} attached."
    if visual.patch_paths:
        return f" {len(visual.patch_paths)} retrieved patch images attached."
    return " no image evidence attached."


def _visual_image_paths(visual: VisualBundle | None) -> list:
    if visual is None:
        return []
    paths = []
    if visual.thumbnail_path:
        paths.append(visual.thumbnail_path)
    paths.extend(visual.patch_paths)
    return paths


def _first_token_prob(choice) -> float:
    try:
        return float(2.718281828 ** choice.logprobs.content[0].logprob)
    except (AttributeError, IndexError, TypeError):
        return 1.0


class DummyBackend:
    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]:
        if node.options:
            return node.options[0], 1.0
        if node.edges:
            return next(iter(node.edges)), 1.0
        return "Sample pathology report.", 1.0


class FineTunedBackend:
    """LoRA-tuned Qwen3-VL node answerer, served locally via HF ``generate``.

    Matches ``ZeroShotQwenBackend`` (``answer`` + ``complete_json``) so it works with
    ``--structured-answer`` / ``--node-react``. Prompts are built with the same
    ``build_chat_messages`` helper used to construct the training data, so serving
    mirrors training exactly. For higher throughput, merge the adapter
    (``training/merge_lora.py``) and serve with vLLM + ``ZeroShotQwenBackend`` instead.
    """

    def __init__(
        self,
        base_model: str,
        adapter_dir: str | None = None,
        *,
        max_new_tokens: int = 128,
        max_pixels: int = 768 * 28 * 28,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code=True, max_pixels=max_pixels
        )
        load_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            "device_map": "auto",
        }
        try:
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **load_kwargs)
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(base_model, **load_kwargs)

        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        self.model = model.eval()

    def _generate(self, system: str, user: str, image_paths: list) -> str:
        from PIL import Image

        from training.dataset import build_chat_messages

        imgs = [Image.open(p).convert("RGB") for p in image_paths if Path(p).exists()]
        messages = build_chat_messages(system, user, None, len(imgs))
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[imgs] if imgs else None,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]:
        prompt = _build_answer_prompt(node, visual, memory, extra_context)
        raw = self._generate(SYSTEM_PROMPT, prompt, _visual_image_paths(visual))
        return raw, 1.0

    def complete_json(
        self,
        node: Node,
        visual: VisualBundle | None,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any], float, str]:
        raw = self._generate(system_prompt, user_prompt, _visual_image_paths(visual))
        return _parse_json_soft(raw), 1.0, raw
