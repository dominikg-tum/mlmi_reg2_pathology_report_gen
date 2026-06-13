"""Run full diagnostic agent with ablation flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.agent_runner import run_agent_traversal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["dummy", "qwen"], default="dummy")
    parser.add_argument("--memory", default="flat")
    parser.add_argument("--visual", default="thumbnail")
    parser.add_argument("--retriever", default="none")
    parser.add_argument("--navigator", default="graph_guided")
    parser.add_argument("--slide-id", default="")
    parser.add_argument("--search-all-patches", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run_agent_traversal(
        backend=args.backend,
        memory=args.memory,
        visual=args.visual,
        retriever=args.retriever,
        navigator=args.navigator,
        slide_id=args.slide_id,
        search_all_patches=args.search_all_patches,
    )
    text = json.dumps(result.chain, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
