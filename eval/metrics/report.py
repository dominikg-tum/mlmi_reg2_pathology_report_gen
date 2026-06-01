"""Report metrics: ROUGE-L, BLEU-4."""

from __future__ import annotations


def rouge_l(pred_report: str, gt_report: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(gt_report, pred_report)["rougeL"].fmeasure
    except ImportError:
        return _token_f1(pred_report, gt_report)


def bleu4(pred_report: str, gt_report: str) -> float:
    try:
        import sacrebleu

        return sacrebleu.sentence_bleu(pred_report, [gt_report]).score / 100.0
    except ImportError:
        return _token_f1(pred_report, gt_report)


def clinical_accuracy_placeholder(pred_report: str, gt_report: str) -> float:
    """Manual review hook — returns token overlap until clinical scorer exists."""
    return _token_f1(pred_report, gt_report)


def _token_f1(pred: str, gt: str) -> float:
    p = set(pred.lower().split())
    g = set(gt.lower().split())
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
