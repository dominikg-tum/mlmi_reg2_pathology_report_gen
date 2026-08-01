"""Reasoning chain metrics: path validity, Edge-F1, MESS, diagnosis accuracy."""

from __future__ import annotations

from typing import Optional

from eval.metrics.normalize import normalize_answer
from eval.schemas import CaseRecord, ChainStep

# Leaf reporting node — Dominik: exclude from BPV; path should end at diagnosis.
REPORT_NODE_ID = "report"
DIAGNOSIS_NODE_ID = "diagnosis"

# Graph ``diagnosis`` answer_key → team 6-way final disease label (Xun / Dominik).
# Provisional mapping until GT is stamped with the 6-way taxonomy directly.
DIAGNOSIS_KEY_TO_LABEL = {
    "malignant": "malignant_tumor",
    "premalignant": "precancerous_lesion",
    "benign": "benign_tumor",
    "non_neoplastic": "inflammatory_or_reactive",
    "descriptive": "insufficient_or_uncertain",
}

FINAL_DIAGNOSIS_LABELS = (
    "normal_or_no_significant_abnormality",
    "benign_tumor",
    "inflammatory_or_reactive",
    "malignant_tumor",
    "insufficient_or_uncertain",
    "precancerous_lesion",
)


def cot_steps(record: CaseRecord, *, exclude_report: bool = True) -> list[ChainStep]:
    """Chain steps used for CoT metrics (optionally drop the report node)."""
    if not exclude_report:
        return list(record.chain)
    return [s for s in record.chain if s.node_id != REPORT_NODE_ID]


def cot_node_path(record: CaseRecord, *, exclude_report: bool = True) -> list[str]:
    path = [n for n in record.node_path if n]
    if not path:
        path = [s.node_id for s in record.chain if s.node_id]
    if exclude_report:
        path = [n for n in path if n != REPORT_NODE_ID]
    return path


def _edges(record: CaseRecord, *, exclude_report: bool = True) -> list[tuple[str, str]]:
    out = []
    for s in cot_steps(record, exclude_report=exclude_report):
        key = s.node_id or s.question
        out.append((key, normalize_answer(s.answer)))
    return out


def node_accuracy(pred: CaseRecord, gt: CaseRecord, *, exclude_report: bool = True) -> float:
    gt_map = {
        s.node_id: normalize_answer(s.answer)
        for s in cot_steps(gt, exclude_report=exclude_report)
        if s.node_id
    }
    pred_map = {
        s.node_id: normalize_answer(s.answer)
        for s in cot_steps(pred, exclude_report=exclude_report)
        if s.node_id
    }
    if not gt_map:
        return 0.0
    correct = sum(1 for nid, a in gt_map.items() if pred_map.get(nid) == a)
    return correct / len(gt_map)


def binary_path_validity(
    pred: CaseRecord, gt: CaseRecord, *, exclude_report: bool = True
) -> float:
    """Exact node-path match. By default drops ``report`` so the last node is diagnosis."""
    gp = cot_node_path(gt, exclude_report=exclude_report)
    pp = cot_node_path(pred, exclude_report=exclude_report)
    if gp and pp:
        return 1.0 if pp == gp else 0.0
    pe = _edges(pred, exclude_report=exclude_report)
    ge = _edges(gt, exclude_report=exclude_report)
    return 1.0 if pe == ge else 0.0


def edge_f1(
    pred: CaseRecord, gt: CaseRecord, *, exclude_report: bool = True
) -> dict[str, float]:
    ps = set(_edges(pred, exclude_report=exclude_report))
    gs = set(_edges(gt, exclude_report=exclude_report))
    if not ps and not gs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(gs) if gs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def mess_score(pred: CaseRecord, gt: CaseRecord, *, exclude_report: bool = True) -> float:
    """Semantic similarity of CoT answers; uses sentence-transformers if available."""
    pred_text = " ".join(s.answer for s in cot_steps(pred, exclude_report=exclude_report))
    gt_text = " ".join(s.answer for s in cot_steps(gt, exclude_report=exclude_report))
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


def diagnosis_answer_key(record: CaseRecord) -> str:
    """Raw ``diagnosis`` node answer_key (normalized), or empty if missing."""
    for s in record.chain:
        if s.node_id == DIAGNOSIS_NODE_ID:
            return normalize_answer(s.answer)
    return ""


def map_diagnosis_to_label(answer_key: str) -> str:
    """Map graph diagnosis key → 6-way team label; empty if unknown."""
    key = normalize_answer(answer_key)
    if key in FINAL_DIAGNOSIS_LABELS:
        return key
    return DIAGNOSIS_KEY_TO_LABEL.get(key, "")


def diagnosis_label_accuracy(pred: CaseRecord, gt: CaseRecord) -> Optional[float]:
    """1.0/0.0 if both sides map to a 6-way label; None if GT has no mappable diagnosis."""
    gt_label = map_diagnosis_to_label(diagnosis_answer_key(gt))
    if not gt_label:
        return None
    pred_label = map_diagnosis_to_label(diagnosis_answer_key(pred))
    return 1.0 if pred_label == gt_label else 0.0
