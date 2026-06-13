from eval.edge_parser import chain_dict_to_record, record_to_eval_dict


def test_chain_dict_to_record():
    chain = {
        "slide_id": "CASE.svs",
        "chain-of-thought": [
            {
                "node_id": "organ_procedure",
                "question": "Organ?",
                "answer": "uterus_hysterectomy",
                "next_question": "Compartment?",
            },
            {
                "node_id": "compartment",
                "question": "Compartment?",
                "answer": "endometrium",
                "next_question": "",
            },
        ],
        "node_path": ["organ_procedure", "compartment"],
    }
    record = chain_dict_to_record(chain, report="Final CAP report.")
    assert record.slide_id == "CASE.svs"
    assert len(record.chain) == 2
    assert record.chain[0].node_id == "organ_procedure"
    assert record.report == "Final CAP report."
    assert record.node_path == ["organ_procedure", "compartment"]

    out = record_to_eval_dict(record)
    assert out["slide_id"] == "CASE.svs"
    assert out["report"] == "Final CAP report."
    assert out["chain-of-thought"][0]["node_id"] == "organ_procedure"
