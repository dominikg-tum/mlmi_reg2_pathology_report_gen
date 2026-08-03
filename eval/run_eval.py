"""CLI: score predictions vs ground-truth chains + reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from eval.metrics.chain import (
    binary_path_validity,
    edge_f1,
    final_diagnosis_accuracy,
    mess_score,
    node_accuracy,
)
from eval.metrics.report import (
    bert_score_f1,
    bleu4,
    clinical_accuracy_tokenf1,
    negation_consistency,
    numeric_fidelity,
    rouge_l,
)
from eval.schemas import CaseRecord

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> dict[str, CaseRecord]:
    by_id: dict[str, CaseRecord] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rec = CaseRecord.from_dict(raw)
            key = rec.slide_id or str(len(by_id))
            by_id[key] = rec
    return by_id


def select_eval_keys(
    preds: dict[str, CaseRecord],
    gts: dict[str, CaseRecord],
    *,
    split: str = "",
) -> list[str]:
    """Return slide_ids present in both pred and gt, optionally filtered by gt split."""
    keys = sorted(set(preds) & set(gts))
    if not split:
        return keys
    return [k for k in keys if gts[k].split == split]


def score_pairs(
    preds: dict[str, CaseRecord],
    gts: dict[str, CaseRecord],
    keys: list[str],
    *,
    exclude_report: bool = True,
    skip_bert: bool = False,
) -> dict[str, Any]:
    """Aggregate CoT + report metrics. CoT path metrics drop ``report`` by default."""
    bpvs: list[float] = []
    f1s: list[float] = []
    messes: list[float] = []
    node_accs: list[float] = []
    diag_accs: list[float] = []
    rouges: list[float] = []
    bleus: list[float] = []
    clin: list[float] = []
    berts: list[float] = []
    num_fid: list[float] = []
    neg_score: list[float] = []
    neg_count: list[int] = []

    for k in keys:
        p, g = preds[k], gts[k]
        bpvs.append(binary_path_validity(p, g, exclude_report=exclude_report))
        f1s.append(edge_f1(p, g, exclude_report=exclude_report)["f1"])
        messes.append(mess_score(p, g, exclude_report=exclude_report))
        node_accs.append(node_accuracy(p, g, exclude_report=exclude_report))
        d = final_diagnosis_accuracy(p, g)
        if d is not None:
            diag_accs.append(d)
        rouges.append(rouge_l(p.report, g.report))
        bleus.append(bleu4(p.report, g.report))
        clin.append(clinical_accuracy_tokenf1(p.report, g.report))
        if not skip_bert:
            berts.append(bert_score_f1(p.report, g.report))
        num_fid.append(numeric_fidelity(p.report, g.report))
        score, count = negation_consistency(p.report, g.report)
        neg_score.append(score)
        neg_count.append(count)

    n = len(keys)
    total_neg = sum(neg_count)
    neg_mean: Optional[float]
    if total_neg == 0:
        neg_mean = None
    else:
        neg_mean = sum(s * c for s, c in zip(neg_score, neg_count)) / total_neg

    return {
        "n_cases": n,
        "exclude_report_from_cot": exclude_report,
        "cot": {
            "binary_path_validity": sum(bpvs) / n if n else 0.0,
            "edge_f1": sum(f1s) / n if n else 0.0,
            "mess": sum(messes) / n if n else 0.0,
            "node_accuracy": sum(node_accs) / n if n else 0.0,
            "final_diagnosis_accuracy": (
                sum(diag_accs) / len(diag_accs) if diag_accs else None
            ),
            "n_diagnosis_scored": len(diag_accs),
        },
        "report": {
            "rouge_l": sum(rouges) / n if n else 0.0,
            "bleu4": sum(bleus) / n if n else 0.0,
            "clinical_token_f1": sum(clin) / n if n else 0.0,
            "bert_score_f1": (sum(berts) / n if berts else None),
            "numeric_fidelity": sum(num_fid) / n if n else 0.0,
            "negation_consistency": neg_mean,
            "negation_comparable_mentions": total_neg,
        },
        # Flat aliases for older consumers / quick tables
        "binary_path_validity": sum(bpvs) / n if n else 0.0,
        "edge_f1": sum(f1s) / n if n else 0.0,
        "mess": sum(messes) / n if n else 0.0,
        "node_accuracy": sum(node_accs) / n if n else 0.0,
        "final_diagnosis_accuracy": (
            sum(diag_accs) / len(diag_accs) if diag_accs else None
        ),
        "rouge_l": sum(rouges) / n if n else 0.0,
        "bleu4": sum(bleus) / n if n else 0.0,
        "clinical_token_f1": sum(clin) / n if n else 0.0,
        "bert_score_f1": (sum(berts) / n if berts else None),
        "numeric_fidelity": sum(num_fid) / n if n else 0.0,
        "negation_consistency": neg_mean,
    }


def print_metrics(metrics: dict[str, Any], *, split: str = "") -> None:
    n = metrics["n_cases"]
    print(f"Cases: {n}" + (f" (split={split})" if split else ""))
    print("--- CoT (report node excluded from path metrics)" if metrics.get(
        "exclude_report_from_cot"
    ) else "--- CoT ---")
    cot = metrics["cot"]
    print(f"Binary Path Validity: {cot['binary_path_validity']:.4f}")
    print(f"Edge-F1:              {cot['edge_f1']:.4f}")
    print(f"MESS:                 {cot['mess']:.4f}")
    print(f"Node Accuracy:        {cot['node_accuracy']:.4f}")
    d = cot["final_diagnosis_accuracy"]
    if d is None:
        print("Final Diag Acc: N/A")
    else:
        print(
            f"Final Diag Acc:         {d:.4f}  "
            f"(n={cot['n_diagnosis_scored']})"
        )
    print("--- Report ---")
    rep = metrics["report"]
    print(f"ROUGE-L:              {rep['rouge_l']:.4f}")
    print(f"BLEU-4:               {rep['bleu4']:.4f}")
    print(f"Clinical (proxy):     {rep['clinical_token_f1']:.4f}")
    if rep["bert_score_f1"] is None:
        print("BERT:                 skipped")
    else:
        print(f"BERT:                 {rep['bert_score_f1']:.4f}")
    print(f"Num. FID:             {rep['numeric_fidelity']:.4f}")
    if rep["negation_consistency"] is None:
        print("Negation Consistency:  N/A (no comparable keywords found)")
    else:
        print(
            f"Negation Consistency:  {rep['negation_consistency']:.4f}  "
            f"(over {rep['negation_comparable_mentions']} comparable mentions)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="REG² eval harness")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Evaluate only GT records with this split (e.g. test).",
    )
    parser.add_argument(
        "--include-report-node",
        action="store_true",
        help="Include the report node in CoT path / Edge-F1 / node accuracy (legacy).",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERTScore (faster / lighter installs).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write aggregated metrics JSON (e.g. metrics.json).",
    )
    args = parser.parse_args()

    preds = load_jsonl(args.pred)
    gts = load_jsonl(args.gt)
    keys = select_eval_keys(preds, gts, split=args.split)
    if not keys:
        if args.split:
            raise SystemExit(
                f"No overlapping slide_ids between pred and gt for split={args.split!r}"
            )
        raise SystemExit("No overlapping slide_ids between pred and gt")

    metrics = score_pairs(
        preds,
        gts,
        keys,
        exclude_report=not args.include_report_node,
        skip_bert=args.skip_bert,
    )
    print_metrics(metrics, split=args.split)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **metrics,
            "split": args.split,
            "pred": str(args.pred),
            "gt": str(args.gt),
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
