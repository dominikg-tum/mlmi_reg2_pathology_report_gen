"""CLI: score predictions vs ground-truth chains + reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics.chain import binary_path_validity, edge_f1, mess_score, node_accuracy
from eval.metrics.report import bleu4, clinical_accuracy_tokenf1, rouge_l, bert_score_f1, numeric_fidelity, \
    negation_consistency
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

    bpvs, f1s, messes, rouges, bleus, clin, node_acc, bert, num_fid, neg_score, neg_count = [], [], [], [], [], [], [], [], [], [], []
    for k in keys:
        p, g = preds[k], gts[k]
        bpvs.append(binary_path_validity(p, g))
        f1s.append(edge_f1(p, g)["f1"])
        messes.append(mess_score(p, g))
        rouges.append(rouge_l(p.report, g.report))
        bleus.append(bleu4(p.report, g.report))
        clin.append(clinical_accuracy_tokenf1(p.report, g.report))
        node_acc.append(node_accuracy(p, g))
        bert.append(bert_score_f1(p.report, g.report))
        num_fid.append(numeric_fidelity(p.report, g.report))
        score, count = negation_consistency(p.report, g.report)
        neg_score.append(score)
        neg_count.append(count)

    n = len(keys)
    print(f"Cases: {n}" + (f" (split={args.split})" if args.split else ""))
    print(f"Binary Path Validity: {sum(bpvs) / n:.4f}")
    print(f"Edge-F1:              {sum(f1s) / n:.4f}")
    print(f"MESS:                 {sum(messes) / n:.4f}")
    print(f"ROUGE-L:              {sum(rouges) / n:.4f}")
    print(f"BLEU-4:               {sum(bleus) / n:.4f}")
    print(f"Clinical (proxy):     {sum(clin) / n:.4f}")
    print(f"Node Accuracy:       {sum(node_acc) / n:.4f}")
    print(f"BERT:                 {sum(bert) / n:.4f}")
    print(f"Num. FID:             {sum(num_fid) / n:.4f}")
    print(f"Neg Score:           {sum(neg_score) / n:.4f}")
    print(f"Neg Count:           {sum(neg_count) / n:.4f}")


if __name__ == "__main__":
    main()
