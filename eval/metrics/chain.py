"""Reasoning chain metrics: path validity, Edge-F1, MESS."""

from __future__ import annotations

from eval.schemas import CaseRecord


def _edges(record: CaseRecord) -> list[tuple[str, str]]:
    out = []
    for s in record.chain:
        key = s.node_id or s.question
        out.append((key, s.answer.strip().lower()))
    return out


def binary_path_validity(pred: CaseRecord, gt: CaseRecord) -> float:
    if gt.node_path and pred.node_path:
        return 1.0 if pred.node_path == gt.node_path else 0.0
    pe = _edges(pred)
    ge = _edges(gt)
    return 1.0 if pe == ge else 0.0


def edge_f1(pred: CaseRecord, gt: CaseRecord) -> dict[str, float]:
    ps = set(_edges(pred))
    gs = set(_edges(gt))
    if not ps and not gs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(gs) if gs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def mess_score(pred: CaseRecord, gt: CaseRecord) -> float:
    """Semantic similarity of answers; uses sentence-transformers if available."""
    pred_text = " ".join(s.answer for s in pred.chain)
    gt_text = " ".join(s.answer for s in gt.chain)
    if not pred_text or not gt_text:
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        a = model.encode(pred_text)
        b = model.encode(gt_text)
        import numpy as np

        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        return max(0.0, min(1.0, sim))
    except ImportError:
        pt = set(pred_text.lower().split())
        gt_set = set(gt_text.lower().split())
        if not gt_set:
            return 0.0
        return len(pt & gt_set) / len(gt_set)
