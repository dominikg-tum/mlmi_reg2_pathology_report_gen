"""Streamlit frontend for graph-guided pathology report generation."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from agent.frontend import (
    discover_slides,
    evidence_images,
    generate_phase2_report,
    load_retrieval_log,
    load_saved_report,
    load_saved_run,
    resolve_shared_path,
    run_phase1,
    slide_label,
)
from baselines.agent_runner import (
    default_runs_dir,
    load_paths_config,
    load_vision_cache_root,
)


def _endpoint_online(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url.rstrip('/')}/models", timeout=1.5) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _chain_rows(chain: dict) -> list[dict]:
    rows = []
    for index, step in enumerate(chain.get("chain-of-thought") or [], start=1):
        rows.append(
            {
                "Step": index,
                "Node": step.get("node_id", ""),
                "Question": step.get("question", ""),
                "Answer": step.get("answer", ""),
                "Next question": step.get("next_question", ""),
            }
        )
    return rows


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="REG2 Pathology Agent",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = load_paths_config()
    qwen_cfg = cfg["qwen"]
    configured_cache = load_vision_cache_root()
    cache_root = resolve_shared_path(configured_cache) if configured_cache else None
    runs_dir = resolve_shared_path(default_runs_dir(cfg))
    wsi_data_dir = resolve_shared_path(Path(cfg["cluster"]["data_dir"]))

    st.title("REG2 Pathology Agent")
    st.caption(
        "Graph-guided uterine WSI reasoning with per-node visual evidence and VLM answers."
    )
    run_notice = st.session_state.pop("run_notice", "")
    report_notice = st.session_state.pop("report_notice", "")
    if run_notice:
        st.success(run_notice)
    if report_notice:
        st.success(report_notice)

    with st.sidebar:
        st.header("Run configuration")
        backend = st.selectbox("Answer backend", ["qwen", "dummy"])
        memory = st.selectbox("Memory", ["flat", "hipporag2", "graphrag", "none"])
        visual = st.selectbox(
            "Visual evidence",
            ["patch_retrieve", "thumbnail", "none"],
        )
        retriever_choices = ["graph_guided", "titan_cosine", "none"]
        retriever_default = 0 if visual == "patch_retrieve" else 2
        retriever = st.selectbox(
            "Patch retriever",
            retriever_choices,
            index=retriever_default,
            disabled=visual != "patch_retrieve",
        )
        search_all = st.toggle(
            "Search all patches",
            value=False,
            disabled=visual != "patch_retrieve",
        )
        st.divider()
        st.text_input("Qwen endpoint", value=qwen_cfg["api_base_url"], disabled=True)
        online = _endpoint_online(qwen_cfg["api_base_url"]) if backend == "qwen" else True
        if online:
            st.success("Backend available")
        else:
            st.warning("Qwen server is not responding")

    @st.cache_data(ttl=60, show_spinner=False)
    def cached_slides(cache: str, runs: str) -> list:
        return discover_slides(
            cache_root=Path(cache) if cache else None,
            runs_dir=Path(runs),
        )

    slide_options = cached_slides(str(cache_root or ""), str(runs_dir))
    option_by_id = {option.slide_id: option for option in slide_options}

    select_col, action_col = st.columns([3, 1])
    with select_col:
        if slide_options:
            slide_id = st.selectbox(
                "Slide",
                [option.slide_id for option in slide_options],
                format_func=lambda value: slide_label(option_by_id[value]),
            )
        else:
            slide_id = st.text_input("Slide ID", placeholder="TUM_Uterus_0001.svs")
    with action_col:
        st.write("")
        st.write("")
        run_clicked = st.button(
            "Run reasoning chain",
            type="primary",
            use_container_width=True,
            disabled=not slide_id or (backend == "qwen" and not online),
        )

    selected = option_by_id.get(slide_id)
    preview_col, summary_col = st.columns([1, 2])
    with preview_col:
        st.subheader("Slide overview")
        if selected and selected.thumbnail_path and selected.thumbnail_path.exists():
            st.image(
                str(selected.thumbnail_path),
                caption=selected.thumbnail_path.name,
                use_column_width=True,
            )
        else:
            st.info("No thumbnail is available for this slide.")

    saved_chain = load_saved_run(runs_dir, slide_id) if slide_id else None
    with summary_col:
        st.subheader("Agent status")
        status_top = st.columns(2)
        status_top[0].metric("Model", "Qwen3-VL 8B" if backend == "qwen" else "Dummy")
        status_top[1].metric("Visual mode", visual.replace("_", " "))
        status_bottom = st.columns(2)
        status_bottom[0].metric(
            "Cached WSI", "Yes" if selected and selected.has_cache else "No"
        )
        status_bottom[1].metric(
            "Saved chain",
            f"{len(saved_chain.get('node_path', []))} steps" if saved_chain else "None",
        )
        st.write(
            "The execution graph chooses the next question. The VLM only answers the "
            "current node using the attached overview and retrieved pathology patches."
        )

    if run_clicked:
        effective_retriever = retriever if visual == "patch_retrieve" else "none"
        try:
            with st.spinner("Walking the diagnostic graph and querying the VLM..."):
                result, chain_path = run_phase1(
                    slide_id,
                    backend=backend,
                    memory=memory,
                    visual=visual,
                    retriever=effective_retriever,
                    cache_root=cache_root,
                    runs_dir=runs_dir,
                    wsi_data_dir=wsi_data_dir,
                    search_all_patches=search_all,
                )
            st.session_state["chain"] = result.chain
            st.session_state["retrieval_log"] = result.retrieval_log
            st.session_state["chain_path"] = str(chain_path)
            st.session_state["result_slide_id"] = slide_id
            st.session_state.pop("report", None)
            st.session_state["run_notice"] = f"Reasoning chain saved to {chain_path}"
            st.rerun()
        except Exception as exc:
            st.exception(exc)

    session_matches_slide = st.session_state.get("result_slide_id") == slide_id
    chain = st.session_state.get("chain") if session_matches_slide else None
    chain = chain or saved_chain
    retrieval_log = (
        st.session_state.get("retrieval_log") if session_matches_slide else None
    )
    if retrieval_log is None and slide_id:
        retrieval_log = load_retrieval_log(runs_dir, slide_id)

    if not chain:
        st.info("Run the agent or select a slide with an existing chain.")
        return

    chain_tab, evidence_tab, report_tab, raw_tab = st.tabs(
        ["Reasoning chain", "Visual evidence", "Final report", "Raw output"]
    )

    with chain_tab:
        rows = _chain_rows(chain)
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Step": st.column_config.NumberColumn(width="small"),
                "Node": st.column_config.TextColumn(width="medium"),
                "Question": st.column_config.TextColumn(width="large"),
                "Answer": st.column_config.TextColumn(width="medium"),
                "Next question": st.column_config.TextColumn(width="large"),
            },
        )
        st.caption("Path: " + " -> ".join(chain.get("node_path") or []))

    with evidence_tab:
        images = evidence_images(retrieval_log or [])
        if not images:
            st.info("No saved patch images are associated with this chain.")
        else:
            st.caption(f"{len(images)} retrieved images across the diagnostic path.")
            for offset in range(0, len(images), 3):
                columns = st.columns(3)
                for column, item in zip(columns, images[offset : offset + 3]):
                    similarity = item.get("similarity")
                    score = f" | similarity {similarity:.3f}" if isinstance(similarity, float) else ""
                    column.image(
                        str(item["path"]),
                        caption=(
                            f"{item['node_id']} | {item['zoom_level']} | "
                            f"{item['scale']}{score}"
                        ),
                        use_column_width=True,
                    )

    with report_tab:
        report = (
            st.session_state.get("report") if session_matches_slide else ""
        ) or load_saved_report(runs_dir, slide_id)
        if report:
            st.text_area("Generated pathology report", value=report, height=420)
        else:
            st.info("Phase 1 is complete. Generate the report from the answered chain.")

        max_tokens = st.number_input(
            "Maximum report tokens",
            min_value=128,
            max_value=2048,
            value=1024,
            step=128,
        )
        if st.button("Generate final report", disabled=not chain):
            configured_model = Path(cfg.get("models", {}).get("medgemma_4b", ""))
            model_path = str(resolve_shared_path(configured_model))
            if not model_path:
                st.error("MedGemma path is not configured.")
            else:
                try:
                    with st.spinner("Generating the CAP-style pathology report..."):
                        report, report_path = generate_phase2_report(
                            chain,
                            slide_id=slide_id,
                            runs_dir=runs_dir,
                            model_path=model_path,
                            max_new_tokens=int(max_tokens),
                        )
                    st.session_state["report"] = report
                    st.session_state["report_notice"] = f"Report saved to {report_path}"
                    st.rerun()
                except Exception as exc:
                    st.exception(exc)

    with raw_tab:
        st.code(json.dumps(chain, indent=2), language="json")
        st.download_button(
            "Download chain JSON",
            data=json.dumps(chain, indent=2) + "\n",
            file_name=f"{slide_id}.cot_chain.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
