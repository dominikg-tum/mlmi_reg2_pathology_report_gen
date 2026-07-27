"""Multimodal LoRA fine-tune for the Phase-1 node answerer (Qwen3-VL by default).

Uses HuggingFace ``transformers`` + ``peft`` + ``trl`` (SFTTrainer) with a custom
collator that:
  * applies the model chat template to the (system, user+images, assistant) turns,
  * loads the on-disk patch/thumbnail PNGs referenced by each sample,
  * masks everything except the assistant answer tokens (completion-only loss),
    so the model is only supervised on the ``{answer_key, ...}`` target.

Heavy deps (torch/transformers/peft/trl) are imported lazily so the rest of the
repo (and the unit tests) stay importable without a GPU stack.

Install (inside the cluster enroot env):
    pip install "transformers>=4.57" "peft>=0.13" "trl>=0.12" accelerate \
        datasets bitsandbytes pillow qwen-vl-utils
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.dataset import build_chat_messages, load_chain_samples

# Attention + MLP projections of the language tower. Vision encoder stays frozen.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class LoraTrainConfig:
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    # Optimisation
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 4096
    # Memory / precision
    bf16: bool = True
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    # Bookkeeping
    logging_steps: int = 5
    save_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42
    max_pixels: int = 768 * 28 * 28  # cap Qwen-VL visual tokens per image

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "LoraTrainConfig":
        cfg = cls()
        for key, value in (config or {}).items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
        return cfg


def _load_processor(base_model: str, cfg: LoraTrainConfig):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        base_model, trust_remote_code=True, max_pixels=cfg.max_pixels
    )
    # Right padding for causal LM training; prompt tokens sit at the front so the
    # completion-only prefix mask is a clean slice.
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
    return processor


def _load_model(base_model: str, cfg: LoraTrainConfig):
    import torch
    from transformers import AutoModelForImageTextToText

    quant = None
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        )

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if quant is not None:
        load_kwargs["quantization_config"] = quant
        load_kwargs["device_map"] = "auto"

    # Prefer the exact Qwen3-VL class when available; fall back to the generic
    # image-text-to-text auto class (works for Qwen3-VL / InternVL / Qwen2.5-VL).
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **load_kwargs)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(base_model, **load_kwargs)

    model.config.use_cache = False
    return model


class _MultimodalCollator:
    """Render chat template + load images, then build completion-only labels."""

    def __init__(self, processor, cfg: LoraTrainConfig):
        self.processor = processor
        self.cfg = cfg

    def _load_images(self, paths: list[str]):
        from PIL import Image, PngImagePlugin

        # WSI crops carry the slide's (sometimes multi-MB) ICC profile, which trips
        # PIL's anti-decompression-bomb cap on PNG text chunks. These are trusted
        # local files, so lift the cap enough to decode them.
        PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

        images = []
        for p in paths:
            fp = Path(p)
            if fp.exists():
                images.append(Image.open(fp).convert("RGB"))
        return images

    def __call__(self, examples: list[dict[str, Any]]):
        import torch

        proc = self.processor
        texts: list[str] = []
        per_sample_images: list[list] = []
        prompt_lens: list[int] = []

        for ex in examples:
            image_paths = list(ex.get("images") or [])
            imgs = self._load_images(image_paths)
            n_img = len(imgs)
            per_sample_images.append(imgs)

            full_msgs = build_chat_messages(
                ex.get("system", ""), ex.get("user", ""), ex.get("target", ""), n_img
            )
            prompt_msgs = build_chat_messages(
                ex.get("system", ""), ex.get("user", ""), None, n_img
            )
            texts.append(
                proc.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )
            )
            # Length of the prompt (incl. expanded image tokens + assistant header)
            # so we can mask it out of the loss.
            prompt_text = proc.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_inputs = proc(
                text=[prompt_text],
                images=[imgs] if imgs else None,
                return_tensors="pt",
            )
            prompt_lens.append(int(prompt_inputs["input_ids"].shape[1]))

        batch = proc(
            text=texts,
            images=per_sample_images if any(per_sample_images) else None,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        # Mask padding.
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        # Mask the prompt prefix (system + user + images + assistant header).
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, labels.shape[1])
            labels[i, :plen] = -100
        # Belt-and-braces: never train on image placeholder tokens.
        for tok in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            tid = proc.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                labels[batch["input_ids"] == tid] = -100

        batch["labels"] = labels
        return batch


def train_lora(
    train_jsonl: Path,
    output_dir: Path,
    *,
    base_model: str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Run LoRA SFT and save the adapter to ``output_dir``. Returns ``output_dir``."""
    import torch  # noqa: F401  (fail fast with a clear error if torch is missing)
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    cfg = LoraTrainConfig.from_dict(config)
    train_jsonl = Path(train_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_chain_samples(train_jsonl)
    if not samples:
        raise ValueError(f"No training samples found in {train_jsonl}")
    dataset = [s.to_record() for s in samples]

    processor = _load_processor(base_model, cfg)
    model = _load_model(base_model, cfg)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg.seed,
        report_to="none",
        # Required for multimodal SFT: keep image columns and skip TRL's text-only
        # dataset preparation so our collator receives the raw records.
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_text_field="",
        max_length=cfg.max_seq_length,
    )

    collator = _MultimodalCollator(processor, cfg)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    return output_dir
