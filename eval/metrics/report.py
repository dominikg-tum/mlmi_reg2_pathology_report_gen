"""Report metrics: ROUGE-L, BLEU-4."""

from __future__ import annotations

from eval.metrics.normalize import normalize_answer
import re
from collections import Counter

_bertscore = None
_NUM_UNIT = re.compile(r"\d+\.?\d*\s*(?:mm|cm|µm|mcm|%)?", re.IGNORECASE)
NEGATION_WORDS = {"no", "not", "without", "absent", "negative", "none"}
WINDOW_SIZE = 4  # Word count that gets checked after keyword is found
CLINICAL_KEYWORDS = [
    "malignan",  # malignancy, malignant
    "carcinoma", "adenocarcinoma",
    "atyp",  # atypia, atypias, atypical
    "dysplas",  # dysplasia, dysplastic
    "hyperplas",
    "necro",  # necrosis, necrotic
    "invas",  # invasion, invasive
    "metasta",
    "mitos", "mitotic",
    "instability",  # microsatellite instability
    "polyp",
    "adenomyosis", "endometriosis",
    "leiomyoma", "leiomyosarcoma",
    "inflammation", "inflammatory",
]


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


def clinical_accuracy_tokenf1(pred_report: str, gt_report: str) -> float:
    """Manual review hook — returns token overlap until clinical scorer exists."""
    return _token_f1(pred_report, gt_report)


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        from evaluate import load
        _bertscore = load("bertscore")
    return _bertscore


def bert_score_f1(pred_report: str, gt_report: str) -> float:
    if not pred_report or not gt_report:
        return 0.0
    try:
        bertscore = _get_bertscore()
        result = bertscore.compute(
            predictions=[pred_report], references=[gt_report],
            lang="en", model_type="microsoft/deberta-large-mnli",
        )
        return result["f1"][0]
    except (ImportError, OSError, ValueError):
        return _token_f1(pred_report, gt_report)


def _token_f1(pred: str, gt: str) -> float:
    p = set(normalize_answer(pred).split())
    g = set(normalize_answer(gt).split())
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _has_negation_window(words: list[str], keyword_prefix: str, window: int = WINDOW_SIZE) -> bool:
    negation = False
    for i, word in enumerate(words):
        if word.startswith(keyword_prefix):
            window_words = words[max(0, i - window):i]
            if any(w in NEGATION_WORDS for w in window_words):
                negation = True
    return negation


def _has_free_suffix(words: list[str], keyword_prefix: str) -> bool:
    for i, word in enumerate(words):
        if word.startswith(keyword_prefix) and word != "free":
            if word.endswith("free"):
                return True
    return False


def negation_consistency(pred_report: str, gt_report: str) -> tuple[float, int]:
    pred_words = normalize_answer(pred_report).split()
    gt_words = normalize_answer(gt_report).split()

    comparable = []
    for kw in CLINICAL_KEYWORDS:
        in_pred = any(w.startswith(kw) for w in pred_words)
        in_gt = any(w.startswith(kw) for w in gt_words)
        if in_pred and in_gt:
            pred_neg = _has_negation_window(pred_words, kw) or _has_free_suffix(pred_words, kw)
            gt_neg = _has_negation_window(gt_words, kw) or _has_free_suffix(gt_words, kw)
            comparable.append(pred_neg == gt_neg)

    if not comparable:
        return 1.0, 0
    return sum(comparable) / len(comparable), len(comparable)


def numeric_fidelity(pred_report: str, gt_report: str) -> float:
    norm = lambda s: s.replace(" ", "").lower()
    gt_matches = [norm(m) for m in _NUM_UNIT.findall(gt_report) if m.strip()]
    pred_matches = [norm(m) for m in _NUM_UNIT.findall(pred_report) if m.strip()]
    if not gt_matches:
        return 1.0
    gt_counts = Counter(gt_matches)
    pred_counts = Counter(pred_matches)
    matched = sum(min(gt_counts[n], pred_counts[n]) for n in gt_counts)
    return matched / len(gt_matches)
