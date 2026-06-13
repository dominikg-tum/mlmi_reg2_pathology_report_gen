"""Phase 1: graph traversal with patch retrieval → runs/{slide_id}/cot_chain.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.agent_runner import (
    default_runs_dir,
    run_agent_traversal,
    write_phase1_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 graph traversal (skip report node).")
    parser.add_argument("--slide-id", required=True, help="Slide filename e.g. CASE.svs")
    parser.add_argument("--backend", choices=["dummy", "qwen"], default="qwen")
    parser.add_argument("--memory", default="flat")
    parser.add_argument("--visual", default="patch_retrieve")
    parser.add_argument("--retriever", default="graph_guided")
    parser.add_argument("--navigator", default="graph_guided")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--search-all-patches", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Override cot_chain.json path")
    args = parser.parse_args()

    result = run_agent_traversal(
        backend=args.backend,
        memory=args.memory,
        visual=args.visual,
        retriever=args.retriever,
        navigator=args.navigator,
        slide_id=args.slide_id,
        skip_report_nodes=True,
        search_all_patches=args.search_all_patches,
    )

    runs_dir = args.runs_dir or default_runs_dir()
    out_path = args.output or write_phase1_outputs(result, runs_dir, args.slide_id)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.chain, indent=2) + "\n")

    print(f"Phase 1 complete: {out_path}")
    print(json.dumps(result.chain, indent=2))


if __name__ == "__main__":
    main()
