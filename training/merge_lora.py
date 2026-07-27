"""Merge a trained LoRA adapter into the base weights for vLLM serving.

vLLM can also load LoRA adapters directly (``--enable-lora``), but merging yields a
plain checkpoint you can ``vllm serve`` exactly like the base model, keeping the
existing OpenAI-compatible ``ZeroShotQwenBackend`` inference path unchanged.
"""

from __future__ import annotations

from pathlib import Path


def merge_lora(
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    try:
        from transformers import Qwen3VLForConditionalGeneration

        base = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **load_kwargs)
    except Exception:
        base = AutoModelForImageTextToText.from_pretrained(base_model, **load_kwargs)

    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # Persist the processor/tokenizer next to the merged weights so vLLM finds them.
    processor = AutoProcessor.from_pretrained(
        str(adapter_dir) if (adapter_dir / "preprocessor_config.json").exists() else base_model,
        trust_remote_code=True,
    )
    processor.save_pretrained(str(output_dir))
    return output_dir
