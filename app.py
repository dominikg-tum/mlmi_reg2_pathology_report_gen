"""Streamlit baseline: one image, one VLM, one graph-guided diagnostic chain."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from agent.backends import DummyBackend
from agent.frontend import (
    run_fixed_image_chain,
    run_remote_image_chain,
    safe_upload_name,
    save_baseline_result,
    save_uploaded_image,
)
from baselines.agent_runner import load_paths_config

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs" / "image_baseline"
UPLOADS_DIR = RUNS_DIR / "uploads"


def fetch_models(base_url: str) -> list[str]:
    request = Request(f"{base_url.rstrip('/')}/models")
    try:
        with urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return []
    return [
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]


def chain_without_report(chain: dict) -> list[dict]:
    steps = chain.get("chain-of-thought") or []
    return [step for step in steps if step.get("node_id") != "report"]


def _step_to_dict(step) -> dict:
    if isinstance(step, dict):
        return dict(step)
    return {
        "node_id": getattr(step, "node_id", ""),
        "question": getattr(step, "question", ""),
        "answer": getattr(step, "answer", ""),
        "confidence": getattr(step, "confidence", 0.0),
        "raw_answer": getattr(step, "raw_answer", ""),
        "next_question": getattr(step, "next_question", ""),
    }


def chain_from_steps(
    steps,
    *,
    slide_id: str = "",
    include_report: bool = True,
) -> dict:
    step_dicts = [_step_to_dict(step) for step in steps]
    report = ""
    if include_report and step_dicts and step_dicts[-1].get("node_id") == "report":
        report = str(step_dicts[-1].get("answer", ""))
    return {
        "slide_id": slide_id,
        "chain-of-thought": step_dicts,
        "node_path": [str(step.get("node_id", "")) for step in step_dicts],
        "report": report,
    }


def _dot_escape(value: object) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _shorten(value: object, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def reasoning_graph_dot(
    chain: dict,
    *,
    active_node_id: str = "",
    active_question: str = "",
) -> str:
    steps = chain.get("chain-of-thought") or []
    lines = [
        "digraph reasoning_path {",
        '  graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.12,0.08"];',
        '  edge [color="#64748b", penwidth=1.6, arrowsize=0.8];',
    ]
    previous = ""
    for index, step in enumerate(steps):
        node_name = f"n{index}"
        is_report = step.get("node_id") == "report"
        fill = "#ecfdf5" if not is_report else "#eff6ff"
        border = "#15803d" if not is_report else "#1d4ed8"
        label = (
            f"{index + 1}. {step.get('node_id', '')}\\n"
            f"Q: {_shorten(step.get('question', ''), 96)}\\n"
            f"A: {_shorten(step.get('answer', ''), 96)}"
        )
        lines.append(
            f'  {node_name} [label="{_dot_escape(label)}", fillcolor="{fill}", color="{border}"];'
        )
        if previous:
            lines.append(f"  {previous} -> {node_name};")
        previous = node_name

    if active_node_id:
        active_name = f"n{len(steps)}_active"
        label = f"Running: {active_node_id}\\nQ: {_shorten(active_question, 120)}"
        lines.append(
            f'  {active_name} [label="{_dot_escape(label)}", fillcolor="#fff7ed", color="#ea580c", penwidth=2.2];'
        )
        if previous:
            lines.append(f"  {previous} -> {active_name};")

    if not steps and not active_node_id:
        lines.append(
            '  empty [label="Waiting for the diagnostic chain to start", fillcolor="#f8fafc", color="#94a3b8"];'
        )
    lines.append("}")
    return "\n".join(lines)


def render_reasoning_graph(
    st,
    chain: dict,
    *,
    active_node_id: str = "",
    active_question: str = "",
) -> None:
    st.graphviz_chart(
        reasoning_graph_dot(
            chain,
            active_node_id=active_node_id,
            active_question=active_question,
        ),
        use_container_width=True,
    )


def render_chain(st, chain: dict) -> None:
    steps = chain_without_report(chain)
    columns = st.columns(2)
    for index, step in enumerate(steps):
        question = html.escape(str(step.get("question", "")))
        answer = html.escape(str(step.get("answer", "")))
        raw_answer = str(step.get("raw_answer", "") or "").strip()
        confidence = step.get("confidence")
        node_id = html.escape(str(step.get("node_id", "")))
        details = []
        if confidence is not None:
            try:
                details.append(f"confidence {float(confidence):.2f}")
            except (TypeError, ValueError):
                pass
        if raw_answer and raw_answer != step.get("answer"):
            details.append(f"raw: {html.escape(raw_answer)}")
        detail_html = (
            f'<div class="node-id">{" | ".join(details)}</div>' if details else ""
        )
        with columns[index % 2]:
            st.markdown(
                f"""
                <div class="qa-card">
                  <div class="qa-row">
                    <span class="qa-badge question-badge">Q</span>
                    <div class="qa-text"><strong>{question}</strong></div>
                  </div>
                  <div class="qa-row answer-row">
                    <span class="qa-badge answer-badge">A</span>
                    <div class="qa-text">{answer}</div>
                  </div>
                  <div class="node-id">{node_id}</div>
                  {detail_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="REG2 Image Baseline",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          .block-container {max-width: 1320px; padding-top: 2rem;}
          .qa-card {
            border: 2px solid #173642;
            border-radius: 8px;
            padding: 12px 14px 9px 14px;
            margin-bottom: 14px;
            background: #ffffff;
          }
          .qa-row {display: flex; align-items: flex-start; gap: 10px;}
          .answer-row {margin-top: 8px; padding-top: 8px; border-top: 1px solid #dfe7ea;}
          .qa-badge {
            display: inline-flex; align-items: center; justify-content: center;
            min-width: 30px; height: 30px; border-radius: 50%;
            color: white; font-weight: 700; border: 2px solid #173642;
          }
          .question-badge {background: #bd1616;}
          .answer-badge {background: #0876ba;}
          .qa-text {font-size: 0.98rem; line-height: 1.35; padding-top: 3px;}
          .node-id {font-size: 0.72rem; color: #667780; margin: 8px 0 0 40px;}
          .report-box {
            border: 2px solid #173642; border-radius: 8px;
            padding: 18px 20px; background: #f7fbfc;
            font-size: 1.05rem; line-height: 1.55;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg = load_paths_config()
    default_endpoint = cfg["qwen"]["api_base_url"]
    default_model = cfg["qwen"]["model_name"]

    st.title("REG2 Pathology Image Baseline")
    st.caption(
        "Upload one pathology image. The selected VLM follows the fixed diagnostic "
        "graph, answers each question, and combines the chain into a final report."
    )

    with st.sidebar:
        st.header("VLM connection")
        mode = st.selectbox("Backend", ["Remote VLM", "Dummy smoke test"])
        endpoint = st.text_input(
            "OpenAI-compatible endpoint",
            value=default_endpoint,
            disabled=mode != "Remote VLM",
        )
        models = fetch_models(endpoint) if mode == "Remote VLM" else []
        if models:
            default_index = models.index(default_model) if default_model in models else 0
            model_name = st.selectbox("VLM", models, index=default_index)
            st.success("VLM endpoint available")
        else:
            model_name = st.text_input(
                "Model name",
                value=default_model,
                disabled=mode != "Remote VLM",
            )
            if mode == "Remote VLM":
                st.warning("Endpoint unavailable or no models returned")
        api_key = st.text_input(
            "API key",
            value=cfg["qwen"].get("api_key", "EMPTY"),
            type="password",
            disabled=mode != "Remote VLM",
        )

    input_col, preview_col = st.columns([1, 1])
    with input_col:
        st.subheader("Input")
        uploaded = st.file_uploader(
            "Pathology image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )
        run_clicked = st.button(
            "Run diagnostic chain",
            type="primary",
            use_container_width=True,
            disabled=(
                uploaded is None
                or (mode == "Remote VLM" and (not endpoint.strip() or not model_name.strip()))
            ),
        )
    with preview_col:
        st.subheader("Image preview")
        if uploaded is not None:
            st.image(uploaded, caption=uploaded.name, use_column_width=True)
        else:
            st.info("Upload a PNG, JPEG, or WebP image.")

    if run_clicked and uploaded is not None:
        try:
            image_path = save_uploaded_image(
                uploaded.getvalue(),
                uploaded.name,
                UPLOADS_DIR,
            )
            live_status = st.empty()
            live_graph = st.empty()

            def update_live_graph(event: dict) -> None:
                partial_chain = chain_from_steps(
                    event.get("steps", []),
                    slide_id=safe_upload_name(uploaded.name),
                    include_report=True,
                )
                active_node_id = ""
                active_question = ""
                if event.get("event") == "node_started":
                    active_node_id = str(event.get("node_id", ""))
                    active_question = str(event.get("question", ""))
                    live_status.info(f"Running reasoning step: {active_node_id}")
                elif event.get("event") == "node_finished":
                    live_status.success(
                        f"Finished reasoning step: {event.get('node_id', '')}"
                    )
                with live_graph.container():
                    st.subheader("Live reasoning graph")
                    render_reasoning_graph(
                        st,
                        partial_chain,
                        active_node_id=active_node_id,
                        active_question=active_question,
                    )

            with st.spinner("Following the diagnostic graph with the selected VLM..."):
                if mode == "Dummy smoke test":
                    chain = run_fixed_image_chain(
                        image_path,
                        backend=DummyBackend(),
                        image_id=safe_upload_name(uploaded.name),
                        progress_callback=update_live_graph,
                    )
                else:
                    chain = run_remote_image_chain(
                        image_path,
                        base_url=endpoint,
                        model_name=model_name,
                        api_key=api_key,
                        progress_callback=update_live_graph,
                    )
                output_path = save_baseline_result(
                    chain,
                    output_root=RUNS_DIR,
                    image_name=uploaded.name,
                )
            live_status.success("Diagnostic graph run complete")
            st.session_state["baseline_chain"] = chain
            st.session_state["baseline_image"] = safe_upload_name(uploaded.name)
            st.session_state["baseline_output"] = str(output_path)
        except Exception as exc:
            st.exception(exc)

    current_image = safe_upload_name(uploaded.name) if uploaded is not None else ""
    chain = (
        st.session_state.get("baseline_chain")
        if st.session_state.get("baseline_image") == current_image
        else None
    )
    if not chain:
        return

    st.divider()

    st.subheader("Final pathology report")
    report = html.escape(str(chain.get("report", ""))).replace("\n", "<br>")
    st.markdown(
        f'<div class="report-box">{report or "No report was generated."}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Final reasoning path")
    render_reasoning_graph(st, chain)

    st.subheader("Step outputs")
    render_chain(st, chain)
    st.caption(f"Saved to {st.session_state.get('baseline_output', '')}")

    with st.expander("Raw chain JSON"):
        raw = json.dumps(chain, indent=2)
        st.code(raw, language="json")
        st.download_button(
            "Download JSON",
            data=raw + "\n",
            file_name=f"{Path(current_image).stem}.cot_chain.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
