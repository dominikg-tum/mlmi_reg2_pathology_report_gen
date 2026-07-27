"""Run full diagnostic agent with ablation flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.agent_runner import (
    default_runs_dir,
    resolve_search_all_patches,
    run_agent_traversal,
    write_phase1_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["dummy", "qwen", "finetuned"], default="dummy"
    )
    parser.add_argument("--memory", default="flat")
    parser.add_argument("--visual", default="thumbnail")
    parser.add_argument("--retriever", default="none")
    parser.add_argument("--navigator", default="graph_guided")
    parser.add_argument("--slide-id", default="")
    pool = parser.add_mutually_exclusive_group()
    pool.add_argument(
        "--search-all-patches",
        action="store_true",
        help="Force full 20x pool (default from configs/vision.yaml)",
    )
    pool.add_argument(
        "--kmeans-pool",
        action="store_true",
        help="Ablation: restrict cosine rank to K-means centroid pool (kmeans_k)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.slide_id:
        raise SystemExit("--slide-id is required")

    result = run_agent_traversal(
        backend=args.backend,
        memory=args.memory,
        visual=args.visual,
        retriever=args.retriever,
        navigator=args.navigator,
        slide_id=args.slide_id,
        search_all_patches=resolve_search_all_patches(
            kmeans_pool=args.kmeans_pool,
            search_all_patches=args.search_all_patches,
        ),
    )
    text = json.dumps(result.chain, indent=2)
    runs_dir = args.runs_dir or default_runs_dir()
    out_path = write_phase1_outputs(result, runs_dir, args.slide_id)
    print(f"Wrote {out_path}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
