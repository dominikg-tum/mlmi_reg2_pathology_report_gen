from eval.metrics.chain import (
    binary_path_validity,
    edge_f1,
    final_diagnosis_accuracy,
    node_accuracy,
)
from eval.schemas import CaseRecord, ChainStep


def _rec(path: list[str], answers: dict[str, str], report: str = "") -> CaseRecord:
    steps = [
        ChainStep(question=n, answer=answers.get(n, "x"), node_id=n) for n in path
    ]
    return CaseRecord(slide_id="s", chain=steps, report=report, node_path=list(path))


def test_bpv_ignores_trailing_report_by_default():
    gt = _rec(
        ["organ_procedure", "compartment", "diagnosis", "report"],
        {"organ_procedure": "uterus", "compartment": "endo", "diagnosis": "benign", "report": "text"},
    )
    pred = _rec(
        ["organ_procedure", "compartment", "diagnosis", "report"],
        {"organ_procedure": "uterus", "compartment": "endo", "diagnosis": "benign", "report": "other"},
    )
    assert binary_path_validity(pred, gt) == 1.0


def test_bpv_legacy_includes_report_when_requested():
    gt = _rec(["diagnosis", "report"], {"diagnosis": "benign", "report": "a"})
    pred = _rec(["diagnosis", "report"], {"diagnosis": "benign", "report": "b"})
    assert binary_path_validity(pred, gt, exclude_report=False) == 1.0
    pred2 = _rec(["diagnosis"], {"diagnosis": "benign"})
    assert binary_path_validity(pred2, gt, exclude_report=False) == 0.0
    assert binary_path_validity(pred2, gt, exclude_report=True) == 1.0


def test_edge_f1_and_node_acc_drop_report_answers():
    gt = _rec(
        ["diagnosis", "report"],
        {"diagnosis": "malignant", "report": "long free text"},
    )
    pred = _rec(
        ["diagnosis", "report"],
        {"diagnosis": "malignant", "report": "different free text"},
    )
    assert edge_f1(pred, gt)["f1"] == 1.0
    assert node_accuracy(pred, gt) == 1.0
    assert edge_f1(pred, gt, exclude_report=False)["f1"] < 1.0


def test_final_diagnosis_accuracy_exact_graph_key():
    gt = _rec(["diagnosis"], {"diagnosis": "malignant"})
    pred_ok = _rec(["diagnosis"], {"diagnosis": "malignant"})
    pred_bad = _rec(["diagnosis"], {"diagnosis": "benign"})
    assert final_diagnosis_accuracy(pred_ok, gt) == 1.0
    assert final_diagnosis_accuracy(pred_bad, gt) == 0.0


def test_final_diagnosis_accuracy_parses_json_answer():
    gt = _rec(
        ["diagnosis"],
        {"diagnosis": '{"answer_key": "non_neoplastic", "rationale": "", "confidence": 1.0}'},
    )
    pred = _rec(["diagnosis"], {"diagnosis": "non_neoplastic"})
    assert final_diagnosis_accuracy(pred, gt) == 1.0
