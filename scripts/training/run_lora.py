"""CLI: LoRA fine-tune the Phase-1 node answerer (Qwen3-VL by default).

Run in the training enroot env (transformers>=4.57). Example:

    python -m scripts.training.run_lora \
        --train-jsonl training/samples.jsonl \
        --output-dir "$WORK/lora/qwen3vl-uterus/adapter" \
        --epochs 3 --lr 1e-4 --lora-r 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.agent_runner import load_paths_config
from training.lora import train_lora

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "training" / "samples.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="", help="Defaults to configs/paths.yaml finetuned.base_model")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true", help="QLoRA (lower VRAM)")
    args = parser.parse_args()

    cfg = load_paths_config()
    base_model = (
        args.base_model
        or cfg.get("finetuned", {}).get("base_model")
        or cfg["models"]["qwen3_vl_8b"]
    )

    config = {
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
    }

    out = train_lora(
        args.train_jsonl,
        args.output_dir,
        base_model=base_model,
        config=config,
    )
    print(f"Saved LoRA adapter -> {out}")


if __name__ == "__main__":
    main()
