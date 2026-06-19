"""Run thumbnail baselines on chains.jsonl test split (unified chain + Qwen report)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.agent_runner import (
    default_runs_dir,
    load_paths_config,
    run_agent_traversal,
    write_run_outputs,
)
from data.case_slides import iter_chain_records, primary_wsi_for_baseline
from eval.edge_parser import chain_dict_to_record, record_to_eval_dict
from vision.cache import build_slide_cache, cache_thumbnail_path, resolve_thumbnail_path
from vision.mag_config import load_vision_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAINS = REPO_ROOT / "data" / "labels" / "chains.jsonl"

BASELINE_PRESETS: dict[str, dict] = {
    "a": {"memory": "flat", "memory_k": 5, "subdir": "baseline_a", "pred": "predictions_test_baseline_a.jsonl"},
    "b1": {"memory": "hipporag2", "memory_k": 2, "subdir": "baseline_b1", "pred": "predictions_test_baseline_b1.jsonl"},
    "b2": {"memory": "hybridrag", "memory_k": 5, "subdir": "baseline_b2", "pred": "predictions_test_baseline_b2.jsonl"},
}


def _wsi_exists(data_dir: Path, wsi_slide_id: str) -> bool:
    if not data_dir.is_dir():
        return False
    return any(data_dir.rglob(wsi_slide_id))


def _has_patch_embeddings(cache_root: Path, wsi_slide_id: str, level: str = "20x") -> bool:
    slide_cache = build_slide_cache(cache_root, wsi_slide_id)
    if slide_cache is None:
        return False
    emb_path = slide_cache.embedding_path_for_level(level)
    return emb_path is not None and emb_path.exists()


def _apply_baseline(args: argparse.Namespace) -> None:
    if not args.baseline:
        return
    key = args.baseline.lower()
    if key not in BASELINE_PRESETS:
        raise SystemExit(f"Unknown baseline {args.baseline!r}; choose from {list(BASELINE_PRESETS)}")
    preset = BASELINE_PRESETS[key]
    args.memory = preset["memory"]
    args.memory_k = preset["memory_k"]
    if args.runs_subdir is None:
        args.runs_subdir = preset["subdir"]
    if args.predictions is None:
        args.predictions = default_runs_dir() / preset["pred"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Thumbnail baseline on chains.jsonl. Full traversal includes Qwen report. "
            "Multi-slide cases use one primary WSI (default: corpus block, index 1)."
        )
    )
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--split", default="test", help="chains.jsonl split filter (default: test)")
    parser.add_argument("--baseline", choices=["a", "b1", "b2"], default=None, help="Preset memory + output paths")
    parser.add_argument("--backend", choices=["dummy", "qwen"], default="qwen")
    parser.add_argument("--memory", default="flat")
    parser.add_argument("--memory-k", type=int, default=5, help="Semantic memory top-k (B1 default: 2 via --baseline b1)")
    parser.add_argument("--visual", default="thumbnail")
    parser.add_argument("--retriever", default="none")
    parser.add_argument("--navigator", default="graph_guided")
    parser.add_argument("--primary-index", type=int, default=1, help="WSI index for multi-slide cases")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Base runs directory (default: work_dir/runs)")
    parser.add_argument("--runs-subdir", type=Path, default=None, help="Subdir under runs-dir, e.g. baseline_a")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Process only N cases (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases with cot_chain.json already present")
    parser.add_argument("--force", action="store_true", help="Re-run even if cot_chain.json exists")
    parser.add_argument("--dry-run", action="store_true", help="List cases and primary WSI only")
    parser.add_argument(
        "--list-missing-thumbnails",
        action="store_true",
        help="Print primary WSI ids that need build_thumbnail_cache (one per line)",
    )
    parser.add_argument("--require-thumbnail", action="store_true", default=True)
    parser.add_argument("--no-require-thumbnail", action="store_false", dest="require_thumbnail")
    parser.add_argument(
        "--require-embeddings",
        action="store_true",
        help="Skip cases whose primary WSI lacks patch_embeddings_20x.pt under cache_root",
    )
    args = parser.parse_args()
    _apply_baseline(args)

    if not args.chains.exists():
        raise SystemExit(f"Missing chains file: {args.chains}")

    base_runs = args.runs_dir or default_runs_dir()
    runs_dir = base_runs / args.runs_subdir if args.runs_subdir else base_runs
    predictions_path = args.predictions or (runs_dir / "predictions_test_baseline.jsonl")
    vcfg = load_vision_config()
    cache_root = Path(vcfg["cache_root"]).expanduser()
    cfg = load_paths_config()
    data_dir = Path(cfg["cluster"]["data_dir"])

    if args.memory == "hybridrag":
        from memory.hybridrag import HybridRAGMemory, get_chroma_storage_default

        chroma_path = get_chroma_storage_default()
        if not chroma_path.exists():
            print(f"Building HybridRAG index at {chroma_path} from train split...")
            mem = HybridRAGMemory()
            mem.build_index_from_chains(str(args.chains), split="train")

    cases = list(iter_chain_records(args.chains, split=args.split or None))

    planned: list[tuple[dict, str]] = []
    skipped_no_thumb: list[str] = []
    skipped_no_emb: list[str] = []
    wsi_only: list[str] = []
    for record in cases:
        case_id = record["slide_id"]
        wsi_id = primary_wsi_for_baseline(case_id, index=args.primary_index)
        has_dataset_or_cache = (
            resolve_thumbnail_path(cache_root, wsi_id, vcfg=vcfg) is not None
            or cache_thumbnail_path(cache_root, wsi_id) is not None
        )
        has_wsi = _wsi_exists(data_dir, wsi_id)
        if args.require_thumbnail and not has_dataset_or_cache and not has_wsi:
            skipped_no_thumb.append(f"{case_id} -> {wsi_id}")
            continue
        if args.require_embeddings and not _has_patch_embeddings(cache_root, wsi_id):
            skipped_no_emb.append(f"{case_id} -> {wsi_id}")
            continue
        if has_wsi and not has_dataset_or_cache:
            wsi_only.append(wsi_id)
        planned.append((record, wsi_id))

    if args.limit > 0:
        planned = planned[: args.limit]

    label = args.baseline or args.memory
    print(
        f"Baseline={label!r} split={args.split!r}: {len(cases)} ok chains, "
        f"{len(planned)} runnable, {len(skipped_no_thumb)} missing WSI+thumbnail, "
        f"{len(skipped_no_emb)} missing patch embeddings"
    )
    print(f"Output: {runs_dir}")
    if wsi_only:
        print(
            f"{len(wsi_only)} cases have .svs on disk but no prebuilt thumbnail yet — "
            "run: python -m scripts.vision.build_thumbnail_cache --slide <W.svs>"
        )
    if skipped_no_thumb:
        print("Missing thumbnail (first 5):")
        for line in skipped_no_thumb[:5]:
            print(f"  {line}")
    if skipped_no_emb:
        print("Missing patch embeddings (first 5):")
        for line in skipped_no_emb[:5]:
            print(f"  {line}")

    if args.list_missing_thumbnails:
        for wsi_id in sorted(set(wsi_only)):
            print(wsi_id)
        return

    if args.dry_run:
        for record, wsi_id in planned:
            n = len(record["slide_id"].split(","))
            print(f"  case={record['slide_id']!r}  wsi={wsi_id!r}  n_slides={n}")
        return

    completed: list[str] = []
    for record, wsi_id in planned:
        case_id = record["slide_id"]
        chain_path = runs_dir / case_id / "cot_chain.json"
        if chain_path.exists() and args.skip_existing and not args.force:
            completed.append(case_id)
            continue

        result = run_agent_traversal(
            backend=args.backend,
            memory=args.memory,
            memory_k=args.memory_k,
            visual=args.visual,
            retriever=args.retriever,
            navigator=args.navigator,
            slide_id=case_id,
            wsi_slide_id=wsi_id,
            skip_report_nodes=False,
        )
        write_run_outputs(result, runs_dir, case_id)
        completed.append(case_id)
        has_report = bool(result.chain.get("report"))
        print(f"OK  {case_id}  (wsi={wsi_id}, report={'yes' if has_report else 'no'})")

    pred_lines: list[str] = []
    for case_id in completed:
        chain_path = runs_dir / case_id / "cot_chain.json"
        if not chain_path.exists():
            continue
        chain = json.loads(chain_path.read_text())
        record = chain_dict_to_record(chain)
        pred_lines.append(json.dumps(record_to_eval_dict(record)))

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("\n".join(pred_lines) + ("\n" if pred_lines else ""))
    print(f"Wrote {len(pred_lines)} predictions -> {predictions_path}")
    print(
        "Eval: python -m eval.run_eval "
        f"--pred {predictions_path} --gt {args.chains} --split {args.split}"
    )


if __name__ == "__main__":
    main()
