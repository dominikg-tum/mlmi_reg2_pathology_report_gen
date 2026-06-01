"""WP3: extract a Q->A chain from each english_report using local Qwen (vLLM).

This produces the supervised training data for WP5 AND validates that Han's graph
covers your reports. The graph supplies the questions; the report supplies which
path was taken and what the answers are.

Two output uses:
  * label which graph path a case follows (-> Binary Path Validity / Edge-F1 GT)
  * per-node target answers for LoRA fine-tuning
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openai
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "You are a pathology expert. Answer each question based ONLY on the provided "
    "report. If a fact is not stated, answer 'not mentioned'. Return JSON only, as "
    "a list of {\"q\": ..., \"a\": ...} objects in the order asked."
)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or REPO_ROOT / "configs" / "paths.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)


def build_client(cfg: dict[str, Any]) -> openai.OpenAI:
    qwen = cfg["qwen"]
    return openai.OpenAI(base_url=qwen["api_base_url"], api_key=qwen["api_key"])


def extract_qa(
    client: openai.OpenAI,
    model: str,
    report_text: str,
    questions: list[str],
    *,
    constrain_choices: list[list[str]] | None = None,
) -> str:
    """Ask the model to answer `questions` against `report_text`.

    When `constrain_choices` is given (one choice list per question), pass them to
    vLLM guided decoding so answers stay on-graph. For the open extraction pass we
    usually leave it None and constrain later at agent inference time.
    """
    questions_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    extra_body: dict[str, Any] = {}
    if constrain_choices is not None:
        # Single-question constrained answering; loop per question for multiple.
        extra_body["guided_choice"] = constrain_choices[0]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Report:\n{report_text}\n\nQuestions:\n{questions_block}",
            },
        ],
        temperature=0.0,
        extra_body=extra_body or None,
    )
    return response.choices[0].message.content or ""


def parse_qa(raw: str) -> list[dict[str, str]]:
    data = json.loads(raw)
    if isinstance(data, dict) and "qa_chain" in data:
        return data["qa_chain"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON shape: {type(data)}")


def main() -> None:
    cfg = load_config()
    client = build_client(cfg)
    model = cfg["qwen"]["model_name"]

    sample_report = (
        "Endometrioid adenocarcinoma, FIGO grade 2, with deep myometrial invasion."
    )
    sample_questions = [
        "What organ and procedure does this specimen come from?",
        "What is the glandular/architectural pattern?",
        "Is there myometrial invasion?",
    ]
    raw = extract_qa(client, model, sample_report, sample_questions)
    print(raw)


if __name__ == "__main__":
    main()
