"""LoRA fine-tune Qwen3-VL-8B on the WP3 ChainSample dataset.

Consumes ``samples_train.jsonl`` produced by :mod:`training.dataset` and trains a PEFT
LoRA adapter with the SAME multimodal prompt layout used at inference
(:mod:`training.prompt`), so the fine-tuned model sees train/serve-identical inputs. The
saved adapter is loaded back through :class:`agent.backends.FineTunedBackend`.

Heavy dependencies (torch, transformers, peft) are imported lazily inside
:func:`train_lora`; this module imports cleanly on a laptop with none of them installed,
matching the rest of the training package. Real training runs on the cluster inside
enroot via ``scripts/cluster/train_lora.sh``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.dataset import ChainSample, load_chain_samples
from training.prompt import build_chat_messages, sample_image_paths


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class LoraConfig:
    """Tunable knobs for the LoRA run (sane Qwen3-VL SFT defaults)."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    # Freeze the vision tower + merger; adapt the language model only (v1).
    freeze_vision: bool = True

    # Training schedule
    epochs: float = 2.0
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 5
    save_steps: int = 200
    max_seq_len: int = 4096
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, overrides: dict[str, Any] | None) -> "LoraConfig":
        cfg = cls()
        for key, value in (overrides or {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


# --------------------------------------------------------------------------- #
# dataset wrapper
# --------------------------------------------------------------------------- #
class _ChainDataset:
    """Minimal map-style dataset yielding (messages, image_paths) per sample."""

    def __init__(self, samples: list[ChainSample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]
        return {
            "messages": build_chat_messages(sample, include_target=True),
            "image_paths": sample_image_paths(sample),
        }


def _make_collate_fn(processor, *, max_seq_len: int):
    """Collator: chat-template -> processor -> input_ids/pixel_values + masked labels.

    Labels mask pad and image/vision tokens so loss is computed on real text tokens only.
    (Assistant-only masking is a possible future refinement; full-text SFT on short
    answer keys works well as a v1 baseline.)
    """
    from vision.vlm_messages import _load_rgb_image

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Collect ids to exclude from the loss (image / vision special tokens).
    ignore_ids: set[int] = set()
    for attr in ("image_token_id", "video_token_id", "vision_start_token_id",
                 "vision_end_token_id"):
        tid = getattr(getattr(processor, "config", processor), attr, None)
        if isinstance(tid, int):
            ignore_ids.add(tid)

    def collate(examples: list[dict[str, Any]]):
        import torch

        texts = [
            processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            for ex in examples
        ]
        images: list = []
        for ex in examples:
            for p in ex["image_paths"]:
                images.append(_load_rgb_image(Path(p)))

        batch = processor(
            text=texts,
            images=images or None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        for tid in ignore_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    return collate


# --------------------------------------------------------------------------- #
# main entrypoint
# --------------------------------------------------------------------------- #
def train_lora(
    train_jsonl: Path,
    output_dir: Path,
    *,
    base_model: str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Fine-tune ``base_model`` with LoRA on ``train_jsonl``; return ``output_dir``.

    ``output_dir`` receives the PEFT adapter (``adapter_model.safetensors`` +
    ``adapter_config.json``), the processor, and a ``training_meta.json`` record. Point
    ``agent.backends.FineTunedBackend`` (backend='lora') at this directory for inference.
    """
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoProcessor, Trainer, TrainingArguments

    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as _AutoVLM

    train_jsonl = Path(train_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LoraConfig.from_dict(config)

    samples = load_chain_samples(train_jsonl)
    if not samples:
        raise ValueError(f"No training samples found in {train_jsonl}")
    dataset = _ChainDataset(samples)

    torch_dtype = getattr(torch, cfg.dtype, torch.bfloat16)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = _AutoVLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if cfg.freeze_vision:
        _freeze_vision_tower(model)
    if cfg.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=(cfg.dtype == "bfloat16"),
        fp16=(cfg.dtype == "float16"),
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=_make_collate_fn(processor, max_seq_len=cfg.max_seq_len),
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    _write_meta(output_dir, base_model=base_model, cfg=cfg, n_samples=len(samples))
    print(f"Saved LoRA adapter -> {output_dir}")
    return output_dir


def _freeze_vision_tower(model) -> None:
    """Best-effort freeze of the vision encoder / merger so only the LM adapts."""
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(key in lname for key in ("visual", "vision", "image_encoder", "merger")):
            param.requires_grad = False


def _write_meta(output_dir: Path, *, base_model: str, cfg: LoraConfig, n_samples: int) -> None:
    meta = {
        "base_model": base_model,
        "n_train_samples": n_samples,
        "lora": {
            "r": cfg.r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": cfg.target_modules,
            "freeze_vision": cfg.freeze_vision,
        },
        "schedule": {
            "epochs": cfg.epochs,
            "per_device_batch_size": cfg.per_device_batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "learning_rate": cfg.learning_rate,
        },
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# CLI (runs on the cluster inside enroot)
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-VL on ChainSamples.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default="/mnt/projects/mlmi/reg2/models/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file with LoraConfig overrides",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.config_json and args.config_json.exists():
        overrides.update(json.loads(args.config_json.read_text()))
    cli_map = {
        "epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "learning_rate": args.lr,
        "r": args.rank,
        "lora_alpha": args.lora_alpha,
    }
    overrides.update({k: v for k, v in cli_map.items() if v is not None})

    train_lora(
        args.train_jsonl,
        args.output_dir,
        base_model=args.base_model,
        config=overrides,
    )


if __name__ == "__main__":
    main()
