"""Run full diagnostic agent with ablation flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.backends import DummyBackend, ZeroShotQwenBackend
from agent.controller import chain_to_dict, traverse
from agent.memory import CaseMemory
from vision.cache import SlideCache, slide_cache_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "paths.yaml").open() as f:
        return yaml.safe_load(f)


def slide_cache_for(slide_id: str, cache_root: Path | None) -> SlideCache | None:
    if cache_root is None:
        return None
    d = slide_cache_dir(cache_root, slide_id)
    thumb = d / "thumbnail.png"
    return SlideCache(
        slide_id=slide_id,
        thumbnail_path=thumb if thumb.exists() else None,
        embeddings_low=d / "embeddings_low.pt",
        embeddings_mid=d / "embeddings_mid.pt",
        embeddings_high=d / "embeddings_high.pt",
    )


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

    if args.backend == "dummy":
        backend = DummyBackend()
    else:
        import openai

        cfg = load_config()
        q = cfg["qwen"]
        client = openai.OpenAI(base_url=q["api_base_url"], api_key=q["api_key"])
        backend = ZeroShotQwenBackend(client, q["model_name"])

    mem = CaseMemory.from_config(args.memory)
    slide_cache = slide_cache_for(args.slide_id, cache_root) if args.slide_id else None

    steps = traverse(
        backend,
        case_memory=mem,
        slide_cache=slide_cache,
        visual_method=args.visual,
        retriever_method=args.retriever,
        navigator_method=args.navigator,
        cache_root=cache_root,
    )
    out = chain_to_dict(steps, slide_id=args.slide_id)
    text = json.dumps(out, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
