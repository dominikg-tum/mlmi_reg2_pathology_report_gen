"""CLI: merge a trained LoRA adapter into the base weights for vLLM serving.

    python -m scripts.training.merge_lora \
        --adapter-dir "$WORK/lora/qwen3vl-uterus/adapter" \
        --output-dir  "$WORK/lora/qwen3vl-uterus/merged"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.agent_runner import load_paths_config
from training.merge_lora import merge_lora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-model", default="")
    args = parser.parse_args()

    cfg = load_paths_config()
    ft = cfg.get("finetuned", {})
    base_model = args.base_model or ft.get("base_model") or cfg["models"]["qwen3_vl_8b"]
    adapter_dir = args.adapter_dir or Path(ft["adapter_dir"])
    output_dir = args.output_dir or Path(ft["merged_dir"])

    out = merge_lora(base_model, adapter_dir, output_dir)
    print(f"Merged weights -> {out}")
    print("Serve with: vllm serve", out)


if __name__ == "__main__":
    main()
