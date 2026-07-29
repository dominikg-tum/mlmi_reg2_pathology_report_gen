"""Phase 2: selected SS-LLM chain + TITAN embedding → MedGemma report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REPORT_RULES_ADDENDUM = """Rules:
- State only findings supported by the diagnostic chain and specimen.
- Do not define medical terms or explain basic pathology vocabulary.
- Write observable slide-level results (what is present/absent, grade, extent), not textbook definitions.
- CAP-style concise prose; do not mention answer keys or internal reasoning steps.
"""


class SlideProjector:
    """Linear map slide_emb (1024) → model hidden dim (4096). Untrained by default."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 4096):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._layer = None

    def _ensure_layer(self):
        if self._layer is not None:
            return
        import torch
        import torch.nn as nn

        layer = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        self._layer = layer

    def project(self, slide_emb: np.ndarray) -> np.ndarray:
        self._ensure_layer()
        import torch

        x = torch.from_numpy(np.asarray(slide_emb, dtype=np.float32).reshape(1, -1))
        with torch.inference_mode():
            out = self._layer(x)
        return out.squeeze(0).cpu().numpy()

    def save(self, path: Path) -> None:
        self._ensure_layer()
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._layer.state_dict(), path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        self._ensure_layer()
        import torch

        self._layer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return True


def _chain_summary(chain: dict[str, Any]) -> str:
    steps = chain.get("chain-of-thought") or []
    lines = []
    for step in steps:
        q = step.get("question", "")
        a = step.get("answer", "")
        lines.append(f"- {q} → {a}")
    return "\n".join(lines)


def build_report_prompt(
    chain: dict[str, Any],
    *,
    slide_prefix: np.ndarray | None = None,
) -> str:
    """Build a report prompt from the one chain selected by SS-LLM."""
    sid = str(chain.get("slide_id", "") or "slide")
    summary = _chain_summary(chain)
    prefix_note = ""
    if slide_prefix is not None:
        norm = float(np.linalg.norm(slide_prefix))
        prefix_note = (
            f"\nSlide embedding prefix (dim={slide_prefix.shape[0]}, L2={norm:.2f}) "
            "is available for global context.\n"
        )
    return (
        "You are a board-certified pathologist. Synthesize the selected diagnostic "
        "chain into a structured CAP-format uterine pathology report.\n"
        f"Selected physical slide: {chain.get('selected_slide_id', sid)}\n"
        f"{prefix_note}\n"
        "Diagnostic chain:\n"
        f"{summary}\n\n"
        f"{REPORT_RULES_ADDENDUM}\n"
        "Write the final pathology report:"
    )


class MedGemmaReportBackend:
    """Text-only report writer using medgemma-1.5-4b-it."""

    def __init__(self, model_path: str, *, device: str | None = None):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
            device_map=dev if dev == "cuda" else None,
        )
        if dev != "cuda":
            self._model = self._model.to(dev)
        self._model.eval()

    def generate_report(
        self,
        chain: dict[str, Any],
        *,
        slide_emb: np.ndarray | None = None,
        projector: SlideProjector | None = None,
        max_new_tokens: int = 1024,
    ) -> str:
        self._ensure_loaded()
        import torch

        if slide_emb is not None and projector is not None:
            prefix = projector.project(slide_emb)
        else:
            prefix = None

        prompt = build_report_prompt(chain, slide_prefix=prefix)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        if prompt in text:
            text = text.split(prompt, 1)[-1]
        return text.strip()


def load_cot_chain(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.rstrip() + "\n")


def write_case_chain(path: Path, chain: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chain, indent=2) + "\n")
