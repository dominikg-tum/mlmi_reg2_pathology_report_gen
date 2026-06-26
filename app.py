"""Streamlit baseline: one image, one VLM, one graph-guided diagnostic chain."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from agent.backends import DummyBackend
from agent.frontend import (
    build_uni2_embedding_context,
    load_uni2_summary,
    normalize_openai_base_url,
    run_fixed_image_chain,
    run_remote_image_chain,
    run_uni2_embedding,
    safe_upload_name,
    save_baseline_result,
    save_uploaded_image,
)
from baselines.agent_runner import load_paths_config
from scripts.vision._common import default_cache_root, load_vision_config
from vision.wsi_io import slide_id_from_path

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs" / "image_baseline"
UPLOADS_DIR = RUNS_DIR / "uploads"
LESION_UPLOADS_DIR = RUNS_DIR / "lesion_patches"
UNI2_LEVELS = ("1.25x", "2.5x", "5x", "10x")


def fetch_models(base_url: str) -> list[str]:
    try:
        api_root = normalize_openai_base_url(base_url)
        request = Request(f"{api_root}/models")
        with urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return []
    return [
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]


def chain_without_report(chain: dict) -> list[dict]:
    steps = chain.get("chain-of-thought") or []
    return [step for step in steps if step.get("node_id") != "report"]


def render_chain(st, chain: dict) -> None:
    steps = chain_without_report(chain)
    columns = st.columns(2)
    for index, step in enumerate(steps):
        question = html.escape(str(step.get("question", "")))
        answer = html.escape(str(step.get("answer", "")))
        node_id = html.escape(str(step.get("node_id", "")))
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
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_uni2_summary(st, summary: dict) -> None:
    rows = []
    for item in summary.get("levels", []):
        rows.append(
            {
                "Level": item.get("level", ""),
                "Patches": item.get("n_patches", 0),
                "Dim": item.get("embedding_dim", 0),
                "Patch embeddings": item.get("patch_embeddings_path", ""),
                "Slide embedding": item.get("slide_embedding_path", ""),
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    thumbnail = summary.get("thumbnail_path", "")
    if thumbnail and Path(thumbnail).exists():
        st.image(thumbnail, caption="Generated thumbnail", use_column_width=True)


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
    vcfg = load_vision_config()
    default_endpoint = cfg["qwen"]["api_base_url"]
    default_model = cfg["qwen"]["model_name"]
    default_cache = default_cache_root(vcfg)
    uni_cfg = vcfg.get("uni2", {})
    configured_uni_levels = [
        level for level in uni_cfg.get("levels", UNI2_LEVELS) if level in UNI2_LEVELS
    ] or list(UNI2_LEVELS)

    st.title("REG2 Pathology Image Baseline")
    st.caption(
        "Upload one pathology image. The selected VLM follows the fixed diagnostic "
        "graph, answers each question, and combines the chain into a final report."
    )

    with st.sidebar:
        st.header("VLM connection")
        embedding_method = st.selectbox("Embedding method", ["Thumbnail", "UNI2"])
        mode = st.selectbox("Backend", ["Remote VLM", "Dummy smoke test"])
        endpoint = st.text_input(
            "OpenAI-compatible endpoint",
            value=default_endpoint,
            disabled=mode != "Remote VLM",
        )
        if mode == "Remote VLM" and endpoint.strip():
            try:
                st.caption(f"API root: `{normalize_openai_base_url(endpoint)}`")
            except ValueError:
                pass
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
        uploaded = None
        lesion_uploads = []
        wsi_path_text = ""
        selected_levels = list(UNI2_LEVELS)
        max_patches = int(uni_cfg.get("max_patches_per_level", 128) or 128)
        save_patch_images = bool(uni_cfg.get("save_patch_images", False))
        cache_root = default_cache
        uni_repo_path = Path(uni_cfg.get("repo_path", "/Volumes/Xun/UNI"))
        uni2_weights_path = (
            Path(uni_cfg["weights_path"]).expanduser()
            if uni_cfg.get("weights_path")
            else None
        )
        if embedding_method == "Thumbnail":
            uploaded = st.file_uploader(
                "Pathology image",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=False,
            )
        else:
            wsi_path_text = st.text_input(
                "WSI path",
                placeholder="/path/to/TUM_Uterus_0001.svs",
            )
            lesion_uploads = st.file_uploader(
                "Provided lesion patch image(s)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
            )
            selected_levels = st.multiselect(
                "UNI2 magnifications",
                list(UNI2_LEVELS),
                default=configured_uni_levels,
            )
            cache_root = Path(
                st.text_input("Cache root", value=str(default_cache))
            ).expanduser()
            uni_repo_path = Path(
                st.text_input("UNI repo path", value=str(uni_repo_path))
            ).expanduser()
            weights_default = str(uni2_weights_path) if uni2_weights_path else ""
            weights_text = st.text_input("UNI2 weights path", value=weights_default)
            uni2_weights_path = Path(weights_text).expanduser() if weights_text.strip() else None
            max_patches = st.number_input(
                "Max patches per level",
                min_value=0,
                max_value=200000,
                value=max_patches,
                step=64,
                help="Use 0 for no cap. Start small for a smoke test.",
            )
            save_patch_images = st.checkbox("Save patch PNGs", value=save_patch_images)

        has_input = (
            uploaded is not None
            if embedding_method == "Thumbnail"
            else bool(wsi_path_text.strip()) and bool(lesion_uploads)
        )
        run_clicked = st.button(
            "Run diagnostic chain",
            type="primary",
            use_container_width=True,
            disabled=(
                not has_input
                or (mode == "Remote VLM" and (not endpoint.strip() or not model_name.strip()))
                or (embedding_method == "UNI2" and not selected_levels)
            ),
        )
    with preview_col:
        if embedding_method == "Thumbnail":
            st.subheader("Image preview")
            if uploaded is not None:
                st.image(uploaded, caption=uploaded.name, use_column_width=True)
            else:
                st.info("Upload a PNG, JPEG, or WebP image.")
        else:
            st.subheader("UNI2 artifacts")
            if wsi_path_text.strip():
                wsi_path = Path(wsi_path_text).expanduser()
                summary = load_uni2_summary(cache_root, slide_id_from_path(wsi_path))
                if summary:
                    render_uni2_summary(st, summary)
                else:
                    st.info("No UNI2 summary exists yet for this WSI/cache root.")
                if lesion_uploads:
                    for item in lesion_uploads:
                        st.image(item, caption=f"Provided lesion patch: {item.name}")
            else:
                st.info("Enter a WSI path and upload provided lesion patch image(s).")

    if run_clicked and has_input:
        try:
            if embedding_method == "Thumbnail":
                image_path = save_uploaded_image(
                    uploaded.getvalue(),
                    uploaded.name,
                    UPLOADS_DIR,
                )
                image_name = uploaded.name
                input_key = safe_upload_name(image_name)
                uni_summary = None
                embedding_context = ""
            else:
                wsi_path = Path(wsi_path_text).expanduser()
                if not wsi_path.exists():
                    raise FileNotFoundError(f"WSI not found: {wsi_path}")
                with st.spinner("Generating UNI2 WSI embeddings..."):
                    uni_summary = run_uni2_embedding(
                        svs_path=wsi_path,
                        cache_root=cache_root,
                        repo_path=uni_repo_path,
                        weights_path=uni2_weights_path,
                        levels=list(selected_levels),
                        max_patches=int(max_patches),
                        save_patch_images=bool(save_patch_images),
                    )
                image_path = Path(uni_summary["thumbnail_path"])
                image_name = slide_id_from_path(wsi_path)
                input_key = (
                    safe_upload_name(image_name)
                    + "::"
                    + ",".join(safe_upload_name(item.name) for item in lesion_uploads)
                )
                embedding_context = build_uni2_embedding_context(uni_summary)
                lesion_dir = LESION_UPLOADS_DIR / Path(image_name).stem
                lesion_patch_paths = [
                    save_uploaded_image(item.getvalue(), f"{index:02d}_{item.name}", lesion_dir)
                    for index, item in enumerate(lesion_uploads)
                ]

            with st.spinner("Following the diagnostic graph with the selected VLM..."):
                if mode == "Dummy smoke test":
                    chain = run_fixed_image_chain(
                        image_path,
                        backend=DummyBackend(),
                        image_id=safe_upload_name(image_name),
                        embedding_context=embedding_context,
                        patch_paths=lesion_patch_paths if embedding_method == "UNI2" else None,
                    )
                else:
                    chain = run_remote_image_chain(
                        image_path,
                        base_url=endpoint,
                        model_name=model_name,
                        api_key=api_key,
                        embedding_context=embedding_context,
                        patch_paths=lesion_patch_paths if embedding_method == "UNI2" else None,
                    )
                output_path = save_baseline_result(
                    chain,
                    output_root=RUNS_DIR,
                    image_name=image_name,
                )
            st.session_state["baseline_chain"] = chain
            st.session_state["baseline_image"] = input_key
            st.session_state["baseline_output"] = str(output_path)
            st.session_state["uni2_summary"] = uni_summary
            st.session_state["embedding_context"] = embedding_context
            st.session_state["lesion_patch_paths"] = [
                str(path)
                for path in (lesion_patch_paths if embedding_method == "UNI2" else [])
            ]
        except Exception as exc:
            st.exception(exc)

    if embedding_method == "Thumbnail":
        current_image = safe_upload_name(uploaded.name) if uploaded is not None else ""
    else:
        current_image = (
            safe_upload_name(Path(wsi_path_text).name)
            + "::"
            + ",".join(safe_upload_name(item.name) for item in lesion_uploads)
            if wsi_path_text.strip() and lesion_uploads
            else ""
        )
    chain = (
        st.session_state.get("baseline_chain")
        if st.session_state.get("baseline_image") == current_image
        else None
    )
    if not chain:
        return

    st.divider()
    if st.session_state.get("uni2_summary"):
        with st.expander("UNI2 embedding summary", expanded=False):
            render_uni2_summary(st, st.session_state["uni2_summary"])
    if st.session_state.get("embedding_context"):
        with st.expander("VLM embedding context", expanded=False):
            st.code(st.session_state["embedding_context"], language="text")

    st.subheader("Diagnostic reasoning chain")
    render_chain(st, chain)

    st.subheader("Final pathology report")
    report = html.escape(str(chain.get("report", ""))).replace("\n", "<br>")
    st.markdown(
        f'<div class="report-box">{report or "No report was generated."}</div>',
        unsafe_allow_html=True,
    )
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
