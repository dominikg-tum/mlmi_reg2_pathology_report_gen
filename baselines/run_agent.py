"""Run full diagnostic agent with ablation flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.backends import DummyBackend, ZeroShotQwenBackend
from agent.controller import chain_to_dict, traverse
from agent.memory import CaseMemory
from vision.cache import build_slide_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["dummy", "qwen"], default="dummy")
    parser.add_argument("--memory", default="flat")
    parser.add_argument("--visual", default="thumbnail")
    parser.add_argument("--retriever", default="none")
    parser.add_argument("--navigator", default="graph_guided")
    parser.add_argument("--slide-id", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cache_root = None
    vision_cfg_path = REPO_ROOT / "configs" / "vision.yaml"
    if vision_cfg_path.exists():
        try:
            import yaml

            with vision_cfg_path.open() as f:
                vcfg = yaml.safe_load(f)
                cr = vcfg.get("cache_root", "")
                if cr:
                    cache_root = Path(cr).expanduser()
        except ImportError:
            pass

    cfg = load_config()
    wsi_data_dir = Path(cfg["cluster"]["data_dir"])

    if args.backend == "dummy":
        backend = DummyBackend()
    else:
        import openai

        q = cfg["qwen"]
        client = openai.OpenAI(base_url=q["api_base_url"], api_key=q["api_key"])
        backend = ZeroShotQwenBackend(client, q["model_name"])

    mem = CaseMemory.from_config(args.memory)
    slide_cache = (
        build_slide_cache(cache_root, args.slide_id) if args.slide_id and cache_root else None
    )

    steps = traverse(
        backend,
        case_memory=mem,
        slide_cache=slide_cache,
        visual_method=args.visual,
        retriever_method=args.retriever,
        navigator_method=args.navigator,
        cache_root=cache_root,
        wsi_data_dir=wsi_data_dir,
    )
    out = chain_to_dict(steps, slide_id=args.slide_id)
    text = json.dumps(out, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
