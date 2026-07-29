"""Phase 1: SS-LLM graph traversal per physical WSI → case-level cot_chain.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.agent_runner import (
    default_runs_dir,
    resolve_search_all_patches,
    run_case_phase1,
)
from extraction.case_ids import case_spec_from_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 graph traversal (SS-LLM: all WSIs in case_key)."
    )
    parser.add_argument(
        "--slide-id",
        required=True,
        help="Case key / GT slide_id (comma-separated for multi-WSI)",
    )
    parser.add_argument("--backend", choices=["dummy", "qwen", "finetuned"], default="qwen")
    parser.add_argument("--memory", default="flat")
    parser.add_argument("--visual", default="patch_retrieve")
    parser.add_argument("--retriever", default="graph_guided")
    parser.add_argument("--navigator", default="graph_guided")
    parser.add_argument("--node-react", action="store_true", help="Enable bounded per-node ReAct loop")
    parser.add_argument(
        "--structured-answer",
        action="store_true",
        help="Return Step A JSON only (no ReAct loop)",
    )
    parser.add_argument("--paired-regions", action="store_true")
    parser.add_argument(
        "--react-max-iters",
        type=int,
        default=None,
        help="ReAct iterations per node (default: node_react.max_iters in configs/vision.yaml)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override case-level cot_chain.json path",
    )
    args = parser.parse_args()

    case = case_spec_from_key(args.slide_id)
    runs_dir = args.runs_dir or default_runs_dir()
    out_path = run_case_phase1(
        case,
        runs_dir=runs_dir,
        backend=args.backend,
        memory=args.memory,
        visual=args.visual,
        retriever=args.retriever,
        navigator=args.navigator,
        skip_report_nodes=True,
        node_react=args.node_react,
        structured_answer=args.structured_answer,
        paired_regions=args.paired_regions,
        react_max_iters=args.react_max_iters,
        skip_existing=args.skip_existing,
        search_all_patches=resolve_search_all_patches(
            kmeans_pool=args.kmeans_pool,
            search_all_patches=args.search_all_patches,
        ),
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_path.read_text())
        out_path = args.output

    chain = json.loads(Path(out_path).read_text())
    print(f"Phase 1 complete: {out_path}")
    print(json.dumps(chain, indent=2))


if __name__ == "__main__":
    main()
