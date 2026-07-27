"""Evaluate the Phase-1 node answerer on a held-out samples file.

Runs ONE model (base, or base + LoRA adapter) over a test-split samples.jsonl
built the same way as training data (scripts/cluster/build_training_jsonl.sh with
SPLIT=test), generates an answer per (slide, node), parses ``answer_key`` from the
JSON output, and compares it to the ground-truth answer.

Reuses ``FineTunedBackend`` so the prompt/image pathway is identical to training
and to live inference. Evaluate one config per run (fresh process = clean GPU
memory); run twice (base vs adapter) to compare — see scripts/cluster/eval_lora.sh.

Example:
    python -m scripts.training.eval_finetuned \
        --test-jsonl /mnt/home/dogakonuk/lora/test_samples.jsonl \
        --base-model /mnt/projects/mlmi/reg2/models/Qwen3-VL-8B-Instruct \
        --adapter-dir /mnt/home/dogakonuk/lora/qwen3vl-uterus/adapter \
        --report /mnt/home/dogakonuk/lora/eval_finetuned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_samples(path: Path, *, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="",
        help="LoRA adapter dir; empty = evaluate the base model only",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--report", type=Path, default=None, help="Write JSON report")
    args = parser.parse_args()

    if not args.test_jsonl.is_file():
        raise SystemExit(f"Missing test samples: {args.test_jsonl}")

    samples = _load_samples(args.test_jsonl, limit=args.limit)
    if not samples:
        raise SystemExit(f"No samples in {args.test_jsonl}")

    from agent.backends import FineTunedBackend, _parse_json_soft

    adapter = args.adapter_dir or None
    tag = "finetuned" if adapter else "base"
    print(f"[eval] model={tag} base={args.base_model} adapter={adapter} "
          f"samples={len(samples)}", flush=True)

    backend = FineTunedBackend(
        args.base_model,
        adapter,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
    )

    total = 0
    correct = 0
    by_inter: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    by_node: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    rows: list[dict] = []

    for i, s in enumerate(samples):
        raw = backend._generate(
            s.get("system", ""), s.get("user", ""), list(s.get("images") or [])
        )
        parsed = _parse_json_soft(raw)
        pred = str(parsed.get("answer_key", "")).strip()
        gt = str(s.get("target_answer", "")).strip()
        ok = bool(pred) and pred == gt

        total += 1
        correct += int(ok)
        inter = str(s.get("metadata", {}).get("interaction", ""))
        node = str(s.get("node_id", ""))
        by_inter[inter][1] += 1
        by_inter[inter][0] += int(ok)
        by_node[node][1] += 1
        by_node[node][0] += int(ok)
        rows.append(
            {
                "slide_id": s.get("slide_id", ""),
                "node_id": node,
                "interaction": inter,
                "gt": gt,
                "pred": pred,
                "correct": ok,
                "raw": raw,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"[eval] {i + 1}/{len(samples)} running_acc="
                  f"{correct / total:.3f}", flush=True)

    acc = correct / total if total else 0.0
    print(f"\n[eval] model={tag} OVERALL accuracy = {acc:.4f} ({correct}/{total})")
    print("[eval] per-interaction:")
    for k in sorted(by_inter):
        c, t = by_inter[k]
        print(f"    {k:<16} {c / t:.4f} ({c}/{t})")
    print("[eval] per-node:")
    for k in sorted(by_node):
        c, t = by_node[k]
        print(f"    {k:<32} {c / t:.4f} ({c}/{t})")

    if args.report:
        report = {
            "model": tag,
            "base_model": args.base_model,
            "adapter_dir": adapter or "",
            "overall": {"accuracy": acc, "correct": correct, "total": total},
            "per_interaction": {k: {"correct": v[0], "total": v[1]} for k, v in by_inter.items()},
            "per_node": {k: {"correct": v[0], "total": v[1]} for k, v in by_node.items()},
            "rows": rows,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[eval] wrote report -> {args.report}")


if __name__ == "__main__":
    main()
