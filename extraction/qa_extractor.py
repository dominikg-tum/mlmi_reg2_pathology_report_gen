"""WP3: extract Q->A / chain-of-thought from english_report (DOMI).

Produces supervised labels for eval and LoRA training.
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
    "report. If a fact is not stated, answer 'not mentioned'. Return JSON only as "
    '{"chain-of-thought": [{"question": "...", "answer": "...", "next_question": "..."}]}'
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
    questions_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    extra_body: dict[str, Any] = {}
    if constrain_choices is not None:
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


def questions_from_graph() -> list[str]:
    from graph import GRAPH, ROOT_ID

    out = []
    nid = ROOT_ID
    seen = set()
    while nid and nid not in seen:
        seen.add(nid)
        node = GRAPH[nid]
        out.append(node.question)
        if node.is_leaf:
            break
        if node.options:
            nid = node.edges.get(node.options[0])
        elif node.edges:
            nid = next(iter(node.edges.values()))
        else:
            break
    return out


def main() -> None:
    cfg = load_config()
    client = build_client(cfg)
    model = cfg["qwen"]["model_name"]
    sample_report = (
        "Endometrioid adenocarcinoma, FIGO grade 2, with deep myometrial invasion."
    )
    raw = extract_qa(client, model, sample_report, questions_from_graph())
    print(raw)


if __name__ == "__main__":
    main()
