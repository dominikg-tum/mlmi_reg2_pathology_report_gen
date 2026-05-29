"""WP4: zero-shot baseline = the full agent loop with an UNtrained Qwen backend.

This is the "before" number and, crucially, it exercises the whole pipeline
(graph + controller + retrieval + model) before any fine-tuning. If this runs
end-to-end, fine-tuning is just swapping the AnswerBackend.
"""

from __future__ import annotations

from typing import Any

import openai

from graph.controller import AnswerBackend, Step, traverse
from graph.diagnostic_graph import AnswerType, Node


class ZeroShotQwenBackend(AnswerBackend):
    """Answers each node zero-shot. Uses guided decoding to stay on-graph."""

    def __init__(self, client: openai.OpenAI, model: str):
        self.client = client
        self.model = model

    def answer(self, node: Node, patches, memory: list[Step]) -> tuple[str, float]:
        history = "\n".join(f"Q: {s.question}\nA: {s.answer}" for s in memory)
        prompt = (
            f"You are analyzing a uterine pathology slide.\n{history}\n"
            f"Current question: {node.question}\nAnswer:"
        )
        extra_body: dict[str, Any] = {}
        if node.answer_type in (AnswerType.CATEGORICAL, AnswerType.BOOLEAN):
            # Hard constraint: output MUST be one of the node's valid edge labels.
            extra_body["guided_choice"] = node.options

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            logprobs=True,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        confidence = _first_token_prob(choice)
        return answer, confidence


def _first_token_prob(choice) -> float:
    try:
        return float(2.718281828 ** choice.logprobs.content[0].logprob)
    except (AttributeError, IndexError, TypeError):
        return 1.0


if __name__ == "__main__":
    # Placeholder wiring; point base_url/model at your running Qwen server.
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")
    backend = ZeroShotQwenBackend(client, model="Qwen2.5-7B-Instruct")
    chain = traverse(backend)  # retriever=None for a text-only smoke test
    for step in chain:
        print(step)
