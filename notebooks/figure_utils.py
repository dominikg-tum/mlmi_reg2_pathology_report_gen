"""Shared plotting utilities for baseline evaluation presentation figures."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.case_slides import iter_chain_records, primary_wsi_for_baseline
from eval.edge_parser import chain_dict_to_record
from eval.metrics.chain import binary_path_validity, edge_f1, mess_score
from eval.metrics.report import bleu4, rouge_l
from eval.run_paths import run_timestamp
from scripts.inference.run_test_baseline_batch import _has_patch_embeddings
from vision.cache import load_vision_config, resolve_thumbnail_path

PALETTE = {
    "A (flat thumbnail)": "#4C72B0",
    "B1 (HippoRAG2)": "#DD8452",
    "B2 (HybridRAG)": "#55A868",
    "Patch (k=100 centroids)": "#8172B3",
    "GT (test)": "#333333",
}
BASELINE_ORDER = [
    "A (flat thumbnail)",
    "B1 (HippoRAG2)",
    "B2 (HybridRAG)",
    "Patch (k=100 centroids)",
]
CHAIN_METRICS = ["bpv", "edge_f1", "mess"]
REPORT_METRICS = ["rouge_l", "bleu4"]
METRIC_LABELS = {
    "bpv": "Binary Path Validity",
    "edge_f1": "Edge-F1",
    "mess": "MESS",
    "rouge_l": "ROUGE-L",
    "bleu4": "BLEU-4",
}
KEY_NODES = [
    "organ_procedure",
    "compartment",
    "endometrium_assessment",
    "mass_histologic_type",
    "diagnosis",
]
COMPARTMENT_ORDER = [
    "endometrium",
    "myometrium",
    "mass_lesion",
    "junctional_zone",
    "serosa_perimetrium",
    "not_mentioned",
]
COLOR_OK = "#2E7D32"
COLOR_BAD = "#C62828"
COLOR_GT = "#66BB6A"
COLOR_MISSING = "#BDBDBD"


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["slide_id"]] = rec
    return out


def node_answers(rec: dict) -> dict[str, str]:
    return {
        (s.get("node_id") or s.get("question", "")): s.get("answer", "")
        for s in rec.get("chain-of-thought", [])
    }


def first_divergence(pred: dict, gt: dict) -> tuple[str | None, str | None, str | None]:
    pa, ga = node_answers(pred), node_answers(gt)
    for nid in gt.get("node_path") or list(ga.keys()):
        if nid in pa and pa[nid] != ga.get(nid):
            return nid, ga.get(nid), pa.get(nid)
    for nid in pa:
        if nid not in ga:
            return nid, None, pa[nid]
    return None, None, None


def is_wrong_root(pred: dict, gt: dict) -> bool:
    ga, pa = node_answers(gt), node_answers(pred)
    for nid in ("organ_procedure", "compartment"):
        if ga.get(nid) and pa.get(nid) and ga[nid] != pa[nid]:
            return True
    return False


def case_metrics(pred_raw: dict, gt_raw: dict) -> dict:
    p = chain_dict_to_record(pred_raw)
    g = chain_dict_to_record(gt_raw)
    f1 = edge_f1(p, g)
    div_node, div_gt, div_pred = first_divergence(pred_raw, gt_raw)
    return {
        "bpv": binary_path_validity(p, g),
        "edge_f1": f1["f1"],
        "mess": mess_score(p, g),
        "rouge_l": rouge_l(p.report, g.report),
        "bleu4": bleu4(p.report, g.report),
        "pred_report_len": len(p.report or ""),
        "gt_report_len": len(g.report or ""),
        "first_div_node": div_node,
        "first_div_gt": div_gt,
        "first_div_pred": div_pred,
    }


def resolve_predictions(runs_dir: Path) -> dict[str, Path]:
    mapping = {
        "A (flat thumbnail)": runs_dir / "predictions_test_baseline_a.jsonl",
        "B1 (HippoRAG2)": runs_dir / "predictions_test_baseline_b1.jsonl",
        "B2 (HybridRAG)": runs_dir / "predictions_test_baseline_b2.jsonl",
        "Patch (k=100 centroids)": runs_dir / "predictions_test_baseline_patch_retrieve.jsonl",
    }
    return {k: v for k, v in mapping.items() if v.exists()}


def save(fig: plt.Figure, fig_dir: Path, flat_dir: Path, name: str) -> None:
    path = fig_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    shutil.copy2(path, flat_dir / name)
    plt.close(fig)


def _short_answer(ans: str, max_len: int = 16) -> str:
    ans = (ans or "?").replace("_", " ")
    return ans if len(ans) <= max_len else ans[: max_len - 1] + "…"


def draw_path_row(
    ax: plt.Axes,
    gt: dict,
    pred: dict | None,
    *,
    title: str,
    max_nodes: int = 6,
) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")

    gt_path = (gt.get("node_path") or [])[:max_nodes]
    if not gt_path:
        gt_path = [
            s.get("node_id")
            for s in gt.get("chain-of-thought", [])
            if s.get("node_id")
        ][:max_nodes]
    ga = node_answers(gt)
    pa = node_answers(pred) if pred else ga

    n = max(len(gt_path), 1)
    w = min(1.35, 8.5 / n)
    for i, nid in enumerate(gt_path):
        x = 0.3 + i * (w + 0.15)
        if pred is None:
            color = COLOR_GT
        else:
            g_ans = ga.get(nid)
            p_ans = pa.get(nid)
            if p_ans is None:
                color = COLOR_MISSING
            elif p_ans == g_ans:
                color = COLOR_OK
            else:
                color = COLOR_BAD
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.35),
                w,
                0.55,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor="#333",
                linewidth=0.8,
                alpha=0.92,
            )
        )
        ax.text(
            x + w / 2,
            0.72,
            nid.split("_")[0][:10],
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            0.48,
            _short_answer(pa.get(nid) if pred else ga.get(nid)),
            ha="center",
            va="center",
            fontsize=6,
        )
        if i < len(gt_path) - 1:
            ax.annotate(
                "",
                xy=(x + w + 0.1, 0.62),
                xytext=(x + w + 0.02, 0.62),
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.8),
            )


def plot_paths_best_worst_grid(df, gt_test, preds_map, fig_dir, flat_dir):
    baselines = [b for b in BASELINE_ORDER if b in preds_map]
    fig, axes = plt.subplots(len(baselines), 2, figsize=(14, 2.6 * len(baselines)))
    if len(baselines) == 1:
        axes = np.array([axes])
    for i, bl in enumerate(baselines):
        sub = df[df["baseline"] == bl].sort_values("edge_f1")
        if sub.empty:
            continue
        for j, (sid, kind) in enumerate(
            [(sub.iloc[0]["slide_id"], "worst"), (sub.iloc[-1]["slide_id"], "best")]
        ):
            f1 = float(sub[sub["slide_id"] == sid]["edge_f1"].iloc[0])
            draw_path_row(
                axes[i, j],
                gt_test[sid],
                preds_map[bl][sid],
                title=f"{bl.split('(')[0].strip()} — {kind} F1={f1:.3f} …{sid[-24:]}",
            )
    fig.suptitle(
        "Graph paths — best vs worst per baseline (green=GT match, red=wrong)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "example_paths_best_worst_grid.png")


def plot_example_path_graph_single(slide_id, gt_test, preds_map, df, fig_dir, flat_dir, slug):
    gt = gt_test[slide_id]
    blist = [b for b in BASELINE_ORDER if b in preds_map and slide_id in preds_map[b]]
    fig, axes = plt.subplots(1 + len(blist), 1, figsize=(14, 1.4 * (1 + len(blist))))
    draw_path_row(axes[0], gt, None, title=f"Ground truth …{slide_id[-32:]}")
    for i, bl in enumerate(blist, start=1):
        f1 = float(
            df[(df["baseline"] == bl) & (df["slide_id"] == slide_id)]["edge_f1"].iloc[0]
        )
        draw_path_row(
            axes[i],
            gt,
            preds_map[bl][slide_id],
            title=f"{bl.split('(')[0].strip()} F1={f1:.3f}",
        )
    fig.suptitle("Single-case path comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, f"example_path_graph_{slug}.png")


def plot_node_confusion_paired(preds_map, gt_test, fig_dir, flat_dir):
    for nid in KEY_NODES:
        blist = [b for b in BASELINE_ORDER if b in preds_map]
        if nid in ("organ_procedure", "compartment"):
            blist = [b for b in blist if b != "Patch (k=100 centroids)"]
        fig, axes = plt.subplots(1, len(blist), figsize=(3.8 * len(blist), 4))
        if len(blist) == 1:
            axes = [axes]
        for ax, bl in zip(axes, blist):
            gt_vals, pred_vals = [], []
            for sid, pred in preds_map[bl].items():
                if sid not in gt_test:
                    continue
                ga = node_answers(gt_test[sid]).get(nid)
                if ga is None:
                    continue
                gt_vals.append(ga)
                pred_vals.append(node_answers(pred).get(nid, "?"))
            if not gt_vals:
                ax.set_visible(False)
                continue
            ct = pd.crosstab(pd.Series(gt_vals), pd.Series(pred_vals))
            sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(bl.split("(")[0].strip(), fontsize=9)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        fig.suptitle(f"Paired per-node confusion — {nid}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        save(fig, fig_dir, flat_dir, f"node_confusion_paired_{nid}.png")

    rows = []
    for bl in [b for b in BASELINE_ORDER if b in preds_map]:
        for nid in KEY_NODES:
            correct = total = 0
            for sid, pred in preds_map[bl].items():
                if sid not in gt_test:
                    continue
                ga = node_answers(gt_test[sid]).get(nid)
                if ga is None:
                    continue
                total += 1
                if node_answers(pred).get(nid) == ga:
                    correct += 1
            if total:
                rows.append(
                    {
                        "baseline": bl.split("(")[0].strip(),
                        "node": nid,
                        "accuracy": correct / total,
                    }
                )
    if rows:
        acc_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=acc_df, x="node", y="accuracy", hue="baseline", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Node exact-match accuracy — paired on same test cases")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        save(fig, fig_dir, flat_dir, "node_accuracy_paired_summary.png")


def _load_retrieval_log(runs_dir, slide_id):
    path = runs_dir / "baseline_patch_retrieve" / slide_id / "retrieval_log.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else []


def _winner_routing_snippet(gt, pred_a, pred_winner, winner_bl: str) -> str:
    """Show how the winning baseline fixes A's root errors (no runtime RAG deps)."""
    ga = node_answers(gt)
    pa = node_answers(pred_a)
    pw = node_answers(pred_winner)
    tag = winner_bl.split("(")[0].strip()
    lines = [f"A wrong root → {tag} downstream recovery:\n"]
    for nid in ("organ_procedure", "compartment", "endometrium_assessment", "diagnosis"):
        if not ga.get(nid):
            continue
        a_ok = pa.get(nid) == ga.get(nid)
        w_ok = pw.get(nid) == ga.get(nid)
        lines.append(f"{nid}:")
        lines.append(f"  GT:     {ga.get(nid)}")
        lines.append(f"  A:      {pa.get(nid, '?')}{'' if a_ok else '  ← wrong'}")
        lines.append(f"  {tag}:  {pw.get(nid, '?')}{'  ✓' if w_ok else ''}")
    lines.append(
        "\nNote: B2/Patch train-report text is injected at inference\n"
        "and is not persisted in cot_chain.json."
    )
    return "\n".join(lines)


def _thumbnail_path_for_case(slide_id, cache_root):
    return resolve_thumbnail_path(
        cache_root, primary_wsi_for_baseline(slide_id), vcfg=load_vision_config()
    )


def _patch_images_from_log(log, node_id=None, limit=3):
    paths = []
    for event in log:
        if node_id and event.get("node_id") != node_id:
            continue
        for patch in event.get("patches") or []:
            p = patch.get("patch_path")
            if p and Path(p).exists():
                paths.append(Path(p))
            if len(paths) >= limit:
                return paths
    if not paths:
        for event in log:
            for patch in event.get("patches") or []:
                p = patch.get("patch_path")
                if p and Path(p).exists():
                    paths.append(Path(p))
                if len(paths) >= limit:
                    return paths
    return paths[:limit]


def find_wrong_root_improved_cases(df, gt_test, preds_map, runs_dir=None, top_k=2):
    if "A (flat thumbnail)" not in preds_map:
        return []
    a_df = df[df["baseline"] == "A (flat thumbnail)"].set_index("slide_id")
    candidates = []
    for bl in ("B2 (HybridRAG)", "Patch (k=100 centroids)"):
        if bl not in preds_map:
            continue
        for _, row in df[df["baseline"] == bl].iterrows():
            sid = row["slide_id"]
            if sid not in gt_test or sid not in preds_map["A (flat thumbnail)"]:
                continue
            if not is_wrong_root(preds_map["A (flat thumbnail)"][sid], gt_test[sid]):
                continue
            delta = row["edge_f1"] - a_df.loc[sid, "edge_f1"]
            if delta > 0.05:
                has_log = bool(runs_dir and _load_retrieval_log(runs_dir, sid))
                candidates.append((sid, bl, float(delta), has_log))
    candidates.sort(key=lambda x: (-x[2], -int(x[3])))
    out, seen = [], set()
    # Prefer one Patch case with retrieval_log and one B2 case when possible.
    for bl_pick in ("Patch (k=100 centroids)", "B2 (HybridRAG)"):
        for item in candidates:
            if item[1] != bl_pick or item[0] in seen:
                continue
            seen.add(item[0])
            out.append(item[:3])
            break
    for item in candidates:
        if item[0] not in seen:
            seen.add(item[0])
            out.append(item[:3])
        if len(out) >= top_k:
            break
    return out


def plot_wrong_root_still_improves_panel(
    cases, gt_test, preds_map, df, runs_dir, cache_root, fig_dir, flat_dir
):
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 5 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])
    for r, (sid, winner_bl, delta) in enumerate(cases):
        gt = gt_test[sid]
        pred_a = preds_map["A (flat thumbnail)"][sid]
        pred_winner = preds_map[winner_bl][sid]
        a_f1 = float(
            df[(df["baseline"] == "A (flat thumbnail)") & (df["slide_id"] == sid)][
                "edge_f1"
            ].iloc[0]
        )
        w_f1 = float(
            df[(df["baseline"] == winner_bl) & (df["slide_id"] == sid)]["edge_f1"].iloc[0]
        )
        div_node, _, _ = first_divergence(pred_a, gt)
        winner_tag = winner_bl.split("(")[0].strip()

        ax0 = axes[r, 0]
        thumb = _thumbnail_path_for_case(sid, cache_root)
        ax0.axis("off")
        if thumb and thumb.exists():
            ax0.imshow(mpimg.imread(thumb))
        ax0.set_title("WSI thumbnail (5×)", fontsize=9)

        ax1 = axes[r, 1]
        ax1.axis("off")
        ax1.text(
            0.02,
            0.98,
            _winner_routing_snippet(gt, pred_a, pred_winner, winner_bl),
            va="top",
            fontsize=7,
            family="monospace",
            transform=ax1.transAxes,
        )
        ax1.set_title(f"{winner_tag} routing vs A", fontsize=9)

        ax2 = axes[r, 2]
        ax2.axis("off")
        log = _load_retrieval_log(runs_dir, sid)
        patches = _patch_images_from_log(log, node_id="compartment", limit=3) or (
            _patch_images_from_log(log, limit=3) if log else []
        )
        if patches:
            ax2.imshow(np.hstack([mpimg.imread(p) for p in patches]))
            ax2.set_title(f"Patch evidence @ 20× (n={len(patches)})", fontsize=9)
        else:
            draw_path_row(
                ax2,
                gt,
                pred_winner,
                title=f"{winner_tag} path (green=GT match)",
                max_nodes=5,
            )

        axes[r, 0].set_xlabel(
            f"…{sid[-36:]} | wrong @ {div_node} | A F1={a_f1:.2f} → "
            f"{winner_tag} F1={w_f1:.2f} (Δ={delta:+.2f})",
            fontsize=7,
        )

    fig.suptitle(
        "Wrong-root but downstream improvement\n"
        "(A errs at organ_procedure/compartment; winner recovers downstream)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "wrong_root_still_improves_panel.png")


def _wrap_report_block(label: str, text: str, width: int = 72) -> str:
    lines = [label]
    for para in (text or "").strip()[:500].split("\n"):
        para = para.strip()
        if para:
            lines.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(lines)


def plot_report_text_highlights(df, gt_test, preds_map, fig_dir, flat_dir):
    bl = "A (flat thumbnail)"
    if bl not in preds_map:
        return
    sub = df[df["baseline"] == bl].sort_values("rouge_l", ascending=False).head(3)
    fig, axes = plt.subplots(len(sub), 1, figsize=(14, 4.2 * len(sub)))
    if len(sub) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, sub.iterrows()):
        sid = row["slide_id"]
        ax.axis("off")
        gt_block = _wrap_report_block("GT (narrative):", gt_test[sid]["report"])
        pred_block = _wrap_report_block("Pred (template):", preds_map[bl][sid].get("report", ""))
        body = (
            f"ROUGE-L={row['rouge_l']:.3f}   BLEU-4={row['bleu4']:.3f}   …{sid[-28:]}\n\n"
            f"{gt_block}\n\n{pred_block}"
        )
        ax.text(
            0.02,
            0.98,
            body,
            va="top",
            ha="left",
            fontsize=7,
            family="monospace",
            transform=ax.transAxes,
            linespacing=1.4,
            wrap=True,
        )
    fig.suptitle(
        "Report text — even best matches are short templates vs long GT narratives",
        fontsize=11,
        fontweight="bold",
    )
    plt.subplots_adjust(top=0.94, hspace=0.35)
    save(fig, fig_dir, flat_dir, "report_text_highlights.png")


def plot_chain_mess_highlights(df, gt_test, preds_map, fig_dir, flat_dir):
    bl = "B2 (HybridRAG)" if "B2 (HybridRAG)" in preds_map else "A (flat thumbnail)"
    sub = df[df["baseline"] == bl].sort_values("mess", ascending=False).head(2)
    fig, axes = plt.subplots(1, len(sub), figsize=(7 * len(sub), 5))
    if len(sub) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, sub.iterrows()):
        sid = row["slide_id"]
        ga, pa = node_answers(gt_test[sid]), node_answers(preds_map[bl][sid])
        lines = [f"MESS={row['mess']:.3f}  Edge-F1={row['edge_f1']:.3f}\n"]
        for nid in (gt_test[sid].get("node_path") or [])[:7]:
            mark = "✓" if ga.get(nid) == pa.get(nid) else "✗"
            lines.append(f"{mark} {nid}: GT={ga.get(nid,'?')} | Pred={pa.get(nid,'?')}")
        ax.axis("off")
        ax.text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")
    fig.suptitle("Chain MESS highlights (CoT answers, not report text)", fontsize=10)
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "chain_mess_highlights.png")


def plot_patch_retrieval_manifest(slide_id, runs_dir, label, fig_dir, flat_dir, fname):
    log = _load_retrieval_log(runs_dir, slide_id)
    if not log:
        return
    rows = []
    for event in log:
        for rank, patch in enumerate(event.get("patches") or []):
            rows.append(
                [
                    event.get("node_id", ""),
                    event.get("zoom_level", ""),
                    rank,
                    patch.get("index", ""),
                    str(patch.get("coord", "")),
                    f"{patch.get('similarity', 0):.4f}",
                ]
            )
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1])
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    tbl = ax0.table(
        cellText=rows[:14],
        colLabels=["node", "zoom", "rank", "idx", "coord", "sim"],
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.35)
    ax0.set_title(f"Patch manifest — {label} …{slide_id[-28:]}", fontsize=10)
    ax1 = fig.add_subplot(gs[1])
    # Prefer a morphologic node (positive sim) for the mosaic; fall back to first event.
    morph_nodes = [
        ev.get("node_id")
        for ev in log
        if ev.get("node_id")
        in ("endometrium_assessment", "mass_histologic_type", "compartment", "myometrium_assessment")
    ]
    node = morph_nodes[0] if morph_nodes else log[0].get("node_id")
    paths = _patch_images_from_log(log, node_id=node, limit=5) or _patch_images_from_log(
        log, limit=5
    )
    if paths:
        ax1.imshow(np.hstack([mpimg.imread(p) for p in paths]))
    ax1.axis("off")
    ax1.set_title(f"Retrieved 20× patches @ {node}", fontsize=9)
    fig.text(
        0.5,
        0.01,
        "sim = cosine(TITAN text query, patch embedding). Values near 0 are normal; "
        "negative sim on synthesis/diagnosis nodes means weak query–image alignment "
        "(rank order still selects best-available patches).",
        ha="center",
        fontsize=8,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, fig_dir, flat_dir, fname)


def plot_graph_overview(fig_dir, flat_dir):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, w, h, text, color, sub=""):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#333"
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2 + 0.1,
            text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        if sub:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.25,
                sub,
                ha="center",
                va="center",
                fontsize=7,
                style="italic",
            )

    box(0.5, 2, 2.8, 1.2, "organ_procedure", "#AED6F1", "ROOT thumbnail 5×")
    box(4, 2, 2.8, 1.2, "compartment", "#A9DFBF", "thumbnail 10×")
    box(7.5, 3.2, 2.2, 0.9, "endometrium", "#F9E79F")
    box(7.5, 2, 2.2, 0.9, "myometrium", "#F5CBA7")
    box(7.5, 0.8, 2.2, 0.9, "mass_lesion", "#F1948A")
    ax.text(
        7,
        5.2,
        "Deterministic diagnostic graph (simplified)",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    save(fig, fig_dir, flat_dir, "graph_overview_simplified.png")
    full = REPO / "docs" / "uterus_execution_graph.png"
    if full.exists():
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        ax2.imshow(mpimg.imread(full))
        ax2.axis("off")
        ax2.set_title("Full execution graph (20 nodes)", fontsize=12)
        save(fig2, fig_dir, flat_dir, "graph_overview_full.png")


def plot_chain_metrics_grouped(df, fig_dir, flat_dir):
    main = df[df["baseline"].isin(["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"])]
    long = (
        main.groupby("baseline")[CHAIN_METRICS]
        .mean()
        .reset_index()
        .melt(id_vars="baseline", var_name="metric", value_name="score")
    )
    long["metric"] = long["metric"].map(METRIC_LABELS)
    hue = ["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(
        data=long,
        x="metric",
        y="score",
        hue="baseline",
        hue_order=hue,
        palette=[PALETTE[h] for h in hue],
        ax=ax,
    )
    ax.set_ylim(0, max(0.45, long["score"].max() * 1.2))
    ax.set_title("Chain metrics — test split")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, fig_dir, flat_dir, "chain_metrics_comparison.png")


def plot_compartment_stacked(preds_map, gt_test, fig_dir, flat_dir):
    sources = {"GT (test)": gt_test, **{k: v for k, v in preds_map.items() if k != "Patch (k=100 centroids)"}}
    rows = []
    for label, data in sources.items():
        total = len(data)
        c = Counter()
        for rec in data.values():
            for s in rec.get("chain-of-thought", []):
                if s.get("node_id") == "compartment":
                    c[s.get("answer", "?")] += 1
        for comp, cnt in c.items():
            rows.append(
                {"source": label, "compartment": comp, "count": cnt, "pct": 100 * cnt / total}
            )
    cdf = pd.DataFrame(rows)
    order = [
        s
        for s in ["GT (test)", "A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"]
        if s in cdf["source"].unique()
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    bottom = {s: 0.0 for s in order}
    colors = dict(zip(COMPARTMENT_ORDER, sns.color_palette("Set2", len(COMPARTMENT_ORDER))))
    for comp in COMPARTMENT_ORDER:
        vals = [cdf[(cdf.source == s) & (cdf.compartment == comp)].pct.sum() for s in order]
        if sum(vals) == 0:
            continue
        axes[0].bar(
            order,
            vals,
            bottom=[bottom[s] for s in order],
            label=comp,
            color=colors.get(comp, "#999"),
        )
        for i, s in enumerate(order):
            bottom[s] += vals[i]
    axes[0].set_title("Compartment — 100% stacked")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=7)
    sns.barplot(data=cdf, x="compartment", y="count", hue="source", hue_order=order, ax=axes[1])
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "compartment_stacked_and_counts.png")
    shutil.copy2(
        fig_dir / "compartment_stacked_and_counts.png", flat_dir / "compartment_distribution.png"
    )


def plot_node_heatmap_all_baselines(preds_map, gt_test, fig_dir, flat_dir):
    baselines = [
        b for b in ["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"] if b in preds_map
    ]
    acc = np.zeros((len(baselines), len(KEY_NODES)))
    counts = np.zeros((len(baselines), len(KEY_NODES)), dtype=int)
    for i, bl in enumerate(baselines):
        for j, nid in enumerate(KEY_NODES):
            correct = total = 0
            for sid, pred in preds_map[bl].items():
                if sid not in gt_test:
                    continue
                ga = node_answers(gt_test[sid]).get(nid)
                if ga is None:
                    continue
                total += 1
                correct += int(node_answers(pred).get(nid) == ga)
            acc[i, j] = correct / total if total else np.nan
            counts[i, j] = total
    fig, ax = plt.subplots(figsize=(11, 3.8))
    annot = np.array(
        [
            [f"{acc[i,j]:.0%}\n(n={counts[i,j]})" for j in range(len(KEY_NODES))]
            for i in range(len(baselines))
        ]
    )
    sns.heatmap(
        acc,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=[n.replace("_", "\n") for n in KEY_NODES],
        yticklabels=baselines,
        ax=ax,
    )
    save(fig, fig_dir, flat_dir, "node_accuracy_heatmap_all_baselines.png")
    shutil.copy2(
        fig_dir / "node_accuracy_heatmap_all_baselines.png",
        flat_dir / "node_accuracy_heatmap_A.png",
    )


def plot_edge_f1_scatter(df, fig_dir, flat_dir):
    main = df[df["baseline"].isin(["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, bl in enumerate(["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"]):
        sub = main[main["baseline"] == bl]
        x = np.random.default_rng(42).normal(i, 0.08, len(sub))
        axes[0].scatter(x, sub["edge_f1"], s=45, alpha=0.65, color=PALETTE[bl])
        axes[0].hlines(sub["edge_f1"].mean(), i - 0.35, i + 0.35, colors="k", lw=2)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["A", "B1", "B2"])
    axes[0].set_title("Per-case Edge-F1 variance")
    a_sub = main[main["baseline"] == "A (flat thumbnail)"].sort_values("edge_f1")
    axes[1].bar(
        range(len(a_sub)),
        a_sub["edge_f1"],
        width=1.0,
        color=["#C44E52" if v == 0 else "#4C72B0" for v in a_sub["edge_f1"]],
    )
    axes[1].axhline(a_sub["edge_f1"].mean(), color="k", ls="--")
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "edge_f1_scatter_by_case.png")
    shutil.copy2(fig_dir / "edge_f1_scatter_by_case.png", flat_dir / "edge_f1_violin.png")


def plot_paired_patch_vs_a(df, a_preds, patch_preds, fig_dir, flat_dir):
    shared = sorted(set(a_preds) & set(patch_preds))
    fair = df[
        df["slide_id"].isin(shared)
        & df["baseline"].isin(["A (flat thumbnail)", "Patch (k=100 centroids)"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    pivot = fair.pivot(index="slide_id", columns="baseline", values="edge_f1").dropna()
    lim = [0, max(0.85, pivot.max().max() * 1.05)]
    axes[0].scatter(
        pivot["A (flat thumbnail)"],
        pivot["Patch (k=100 centroids)"],
        s=70,
        c=PALETTE["Patch (k=100 centroids)"],
        alpha=0.75,
    )
    axes[0].plot(lim, lim, "k--", alpha=0.45)
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_title(f"Paired Edge-F1 (n={len(pivot)})")
    means = fair.groupby("baseline")[["edge_f1", "bpv", "mess"]].mean()
    x = np.arange(3)
    axes[1].bar(
        x - 0.175,
        means.loc["A (flat thumbnail)"],
        0.35,
        label="A",
        color=PALETTE["A (flat thumbnail)"],
    )
    axes[1].bar(
        x + 0.175,
        means.loc["Patch (k=100 centroids)"],
        0.35,
        label="Patch",
        color=PALETTE["Patch (k=100 centroids)"],
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Edge-F1", "BPV", "MESS"])
    axes[1].legend()
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "paired_patch_vs_A_before_after.png")
    shutil.copy2(
        fig_dir / "paired_patch_vs_A_before_after.png", flat_dir / "paired_edge_f1_patch_vs_A.png"
    )


def _organ_procedure_crosstab(preds: dict, gt_test: dict) -> pd.DataFrame | None:
    gt_vals, pred_vals = [], []
    for sid in preds:
        if sid not in gt_test:
            continue
        ga = node_answers(gt_test[sid]).get("organ_procedure")
        pa = node_answers(preds[sid]).get("organ_procedure")
        if ga and pa:
            gt_vals.append(ga)
            pred_vals.append(pa)
    if not gt_vals:
        return None
    return pd.crosstab(pd.Series(gt_vals), pd.Series(pred_vals))


def _organ_procedure_accuracy(ct: pd.DataFrame) -> tuple[int, int]:
    """Return (correct, total) on diagonal of organ_procedure crosstab."""
    total = int(ct.values.sum())
    correct = int(np.trace(ct.reindex(index=ct.index, columns=ct.index, fill_value=0).values))
    return correct, total


def plot_organ_procedure_confusion_single(
    preds: dict,
    gt_test: dict,
    *,
    title: str,
    filename: str,
    fig_dir: Path,
    flat_dir: Path,
    subtitle: str = "",
) -> pd.DataFrame | None:
    ct = _organ_procedure_crosstab(preds, gt_test)
    if ct is None:
        return None
    correct, total = _organ_procedure_accuracy(ct)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Cases"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{title}\n{subtitle}".strip())
    ax.text(
        0.02,
        0.98,
        f"accuracy {correct}/{total} ({correct / total:.0%})",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    save(fig, fig_dir, flat_dir, filename)
    return ct


def plot_organ_procedure_confusion_all_baselines(
    preds_map: dict[str, dict], gt_test: dict, fig_dir: Path, flat_dir: Path
) -> None:
    """Per-baseline organ_procedure heatmaps + combined comparison grid."""
    panels: list[tuple[str, dict, str, str]] = []
    for bl, preds in preds_map.items():
        if bl == "Patch (k=100 centroids)":
            shared = sorted(
                set(preds)
                & set(preds_map.get("A (flat thumbnail)", {}))
                & set(gt_test)
            )
            if not shared:
                continue
            sub = {sid: preds[sid] for sid in shared}
            tag = bl.split("(")[0].strip()
            panels.append(
                (
                    f"{tag} (paired n={len(shared)})",
                    sub,
                    f"organ_procedure_confusion_patch_paired.png",
                    f"Fair subset — same cases as A (n={len(shared)})",
                )
            )
            continue
        tag = bl.split("(")[0].strip()
        suffix = {"A (flat thumbnail)": "A", "B1 (HippoRAG2)": "B1", "B2 (HybridRAG)": "B2"}.get(
            bl, tag
        )
        subtitle = ""
        if bl == "A (flat thumbnail)":
            subtitle = "Thumbnail-only root: curettage often misread as hysterectomy"
        elif bl == "B2 (HybridRAG)":
            subtitle = "Report RAG at root — fixes curettage, may over-call curettage"
        panels.append(
            (
                f"{tag} (n={len(preds)})",
                preds,
                f"organ_procedure_confusion_{suffix}.png",
                subtitle,
            )
        )

    cts: list[tuple[str, pd.DataFrame, str]] = []
    for title, preds, filename, subtitle in panels:
        ct = plot_organ_procedure_confusion_single(
            preds,
            gt_test,
            title=f"organ_procedure — {title}",
            filename=filename,
            fig_dir=fig_dir,
            flat_dir=flat_dir,
            subtitle=subtitle,
        )
        if ct is not None:
            cts.append((title, ct, subtitle))

    if len(cts) < 2:
        return

    ncols = min(2, len(cts))
    nrows = (len(cts) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    all_gt = sorted({g for _, ct, _ in cts for g in ct.index})
    all_pred = sorted({p for _, ct, _ in cts for p in ct.columns})
    vmax = max(int(ct.values.max()) for _, ct, _ in cts)

    for ax, (title, ct, subtitle) in zip(axes, cts):
        aligned = ct.reindex(index=all_gt, columns=all_pred, fill_value=0)
        correct, total = _organ_procedure_accuracy(ct)
        sns.heatmap(
            aligned,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            vmin=0,
            vmax=vmax,
            cbar=False,
        )
        ax.set_title(f"{title}\n{correct}/{total} correct", fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(cts) :]:
        ax.set_visible(False)
    fig.suptitle(
        "organ_procedure confusion — all baselines\n"
        "(Patch: paired n=22; A/B1/B2: full test n=69)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "organ_procedure_confusion_all_baselines.png")


def plot_organ_procedure_confusion(preds, gt_test, fig_dir, flat_dir):
    plot_organ_procedure_confusion_single(
        preds,
        gt_test,
        title="organ_procedure confusion — Baseline A",
        filename="organ_procedure_confusion_A.png",
        fig_dir=fig_dir,
        flat_dir=flat_dir,
        subtitle="Thumbnail-only root: curettage often misread as hysterectomy",
    )


def plot_first_divergence(df, fig_dir, flat_dir):
    a_df = df[df["baseline"] == "A (flat thumbnail)"]
    if not a_df.empty:
        div = a_df["first_div_node"].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        div.sort_values().plot(
            kind="barh",
            ax=ax,
            color=["#C44E52" if n == "organ_procedure" else "#4C72B0" for n in div.index],
        )
        ax.set_title("First divergence — Baseline A")
        ax.set_xlabel("Number of test cases")
        save(fig, fig_dir, flat_dir, "first_divergence_node_A.png")

    plot_first_divergence_all_baselines(df, fig_dir, flat_dir)


def plot_first_divergence_all_baselines(df: pd.DataFrame, fig_dir: Path, flat_dir: Path) -> None:
    """Grouped horizontal bars — where each baseline first diverges from GT path."""
    baselines = [b for b in BASELINE_ORDER if b in set(df["baseline"])]
    if not baselines:
        return

    rows: list[dict] = []
    for bl in baselines:
        sub = df[df["baseline"] == bl]
        n_cases = len(sub)
        for node, cnt in sub["first_div_node"].value_counts().items():
            label = "(perfect path)" if node is None else str(node)
            rows.append(
                {
                    "baseline": bl.split("(")[0].strip(),
                    "baseline_full": bl,
                    "node": label,
                    "count": int(cnt),
                    "n": n_cases,
                }
            )
    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    # Top nodes by max count across baselines (exclude perfect-path bucket from ranking)
    node_totals = (
        plot_df[plot_df["node"] != "(perfect path)"]
        .groupby("node")["count"]
        .max()
        .sort_values(ascending=False)
    )
    top_nodes = list(node_totals.head(8).index)
    if (plot_df["node"] == "(perfect path)").any():
        top_nodes.append("(perfect path)")
    plot_df = plot_df[plot_df["node"].isin(top_nodes)]

    node_order = [n for n in reversed(top_nodes) if n != "(perfect path)"]
    if "(perfect path)" in top_nodes:
        node_order = ["(perfect path)"] + node_order

    hue_order = [b.split("(")[0].strip() for b in baselines]
    n_by_bl = df.groupby("baseline").size().to_dict()
    hue_labels = [
        f"{b.split('(')[0].strip()} (n={n_by_bl.get(b, 0)})" for b in baselines
    ]
    label_map = dict(zip(hue_order, hue_labels))
    plot_df["baseline_label"] = plot_df["baseline"].map(label_map)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.barplot(
        data=plot_df,
        y="node",
        x="count",
        hue="baseline_label",
        order=node_order,
        hue_order=hue_labels,
        palette=[PALETTE[b] for b in baselines],
        ax=ax,
        edgecolor="white",
        linewidth=0.6,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2, fontsize=8)

    ax.set_xlabel("Number of test cases where this is the first wrong node")
    ax.set_ylabel("")
    ax.set_title(
        "First divergence from GT path — all baselines\n"
        "(bold red label: organ_procedure = thumbnail-only root)",
        fontweight="bold",
    )
    for tick in ax.get_yticklabels():
        if tick.get_text() == "organ_procedure":
            tick.set_color("#C44E52")
            tick.set_fontweight("bold")

    ax.legend(title="Baseline", loc="lower right", fontsize=9)
    ax.set_xlim(0, max(plot_df["count"].max() * 1.18, 5))
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "first_divergence_all_baselines.png")
    shutil.copy2(
        fig_dir / "first_divergence_all_baselines.png",
        flat_dir / "first_divergence_node_all.png",
    )


def plot_report_metrics(df, fig_dir, flat_dir):
    main = df[df["baseline"].isin(["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"])]
    long = (
        main.groupby("baseline")[REPORT_METRICS]
        .mean()
        .reset_index()
        .melt(id_vars="baseline", var_name="metric", value_name="score")
    )
    long["metric"] = long["metric"].map(METRIC_LABELS)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=long, x="metric", y="score", hue="baseline", ax=ax)
    ax.set_title("Report metrics — ROUGE-L & BLEU-4 only")
    save(fig, fig_dir, flat_dir, "report_metrics_comparison.png")


def plot_report_length(df, fig_dir, flat_dir):
    parts, labels = [df.drop_duplicates("slide_id")["gt_report_len"].values], ["GT"]
    for bl in df["baseline"].unique():
        parts.append(df[df["baseline"] == bl]["pred_report_len"].values)
        labels.append(bl.split("(")[0].strip())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(parts, tick_labels=labels)
    ax.set_title("Report length — template vs narrative GT")
    ax.tick_params(axis="x", rotation=20)
    save(fig, fig_dir, flat_dir, "report_length_boxplot.png")


def plot_rag_effect_summary(df, preds_map, gt_test, fig_dir, flat_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    main = df[df["baseline"].isin(["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"])]
    m = main.groupby("baseline")["edge_f1"].mean()
    axes[0].bar(range(len(m)), m.values, color=[PALETTE[b] for b in m.index])
    axes[0].set_xticks(range(len(m)))
    axes[0].set_xticklabels(["A", "B1", "B2"])
    axes[0].set_title("Mean Edge-F1 by RAG variant")
    for i, v in enumerate(m.values):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")
    comp_rows = []
    for bl in ["A (flat thumbnail)", "B1 (HippoRAG2)", "B2 (HybridRAG)"]:
        if bl not in preds_map:
            continue
        for rec in preds_map[bl].values():
            for s in rec.get("chain-of-thought", []):
                if s.get("node_id") == "compartment":
                    comp_rows.append(
                        {"baseline": bl.split("(")[0].strip(), "compartment": s.get("answer")}
                    )
    if comp_rows:
        pd.crosstab(
            pd.DataFrame(comp_rows)["baseline"], pd.DataFrame(comp_rows)["compartment"]
        ).plot(kind="bar", ax=axes[1], rot=0, legend=True, fontsize=7)
        axes[1].set_title("Compartment by baseline")
    fig.suptitle("RAG effects (refresh after B1 fix rerun)", fontsize=11)
    plt.tight_layout()
    save(fig, fig_dir, flat_dir, "presentation_summary_rag.png")


def plot_patch_coverage(cache_root, chains_path, fig_dir, flat_dir):
    n_ok = sum(
        _has_patch_embeddings(cache_root, primary_wsi_for_baseline(r["slide_id"]))
        for r in iter_chain_records(chains_path, split="test")
    )
    n = sum(1 for _ in iter_chain_records(chains_path, split="test"))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [n_ok, n - n_ok],
        labels=[f"Encoded {n_ok}", f"Missing {n - n_ok}"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title("Patch embedding coverage")
    save(fig, fig_dir, flat_dir, "patch_embedding_coverage.png")


def write_narrative_md(fig_dir, mean_tbl):
    text = """# Presentation narrative — REG² baseline eval (v2)

## Recommended slide order (14 slides)

| # | Slide | Figure file |
|---|-------|-------------|
| 1 | **Problem** — Agent walks a fixed diagnostic graph | `graph_overview_simplified.png` |
| 2 | **Full graph** (backup) — 20 nodes | `graph_overview_full.png` |
| 3 | **Setup** — 69 test cases, GT=text-oracle, pred=vision on 1 WSI | (bullet slide) |
| 4 | **Chain metrics** — BPV, Edge-F1, MESS | `chain_metrics_comparison.png` |
| 5 | **Where A fails** — organ_procedure + first divergence | `organ_procedure_confusion_all_baselines.png`, `first_divergence_all_baselines.png`, `first_divergence_node_A.png` |
| 6 | **Path examples** — best/worst per baseline | `example_paths_best_worst_grid.png` |
| 7 | **Node accuracy** — heatmap + paired confusion | `node_accuracy_heatmap_all_baselines.png`, `node_confusion_paired_*.png` |
| 8 | **RAG story** — B2 helps; B1 fix pending | `presentation_summary_rag.png` |
| 9 | **Wrong root, still improves** — thumbnail vs RAG vs patches | `wrong_root_still_improves_panel.png` |
| 10 | **Report caveat** — genre mismatch | `report_length_boxplot.png`, `report_text_highlights.png` |
| 11 | **Chain MESS** — CoT answers not report | `chain_mess_highlights.png` |
| 12 | **Patch retrieve** — fair n=22 paired | `paired_patch_vs_A_before_after.png` |
| 13 | **Patch manifest** — best/worst cases | `patch_retrieval_manifest_*.png` |
| 14 | **Coverage** — 22/69 encoded | `patch_embedding_coverage.png` |

## Key talking points

- **Show the graph** — `organ_procedure` → `compartment` are thumbnail-only root hops.
- **Do not apologize for low BPV** — strict exact-path metric + early-branch errors.
- **Split report vs chain metrics** — ROUGE ~0.03 is genre mismatch, not null result.
- **Never compare patch n=22 to A n=69** — always paired subset.
- **B1 numbers are stale** — rerun with fix before final presentation.

## Mean metrics (auto-generated)

"""
    try:
        text += mean_tbl.to_markdown() + "\n"
    except ImportError:
        text += mean_tbl.to_string() + "\n"
    for dest in (
        fig_dir / "PRESENTATION_NARRATIVE.md",
        fig_dir.parent / "PRESENTATION_NARRATIVE.md",
        REPO / "notebooks" / "PRESENTATION_NARRATIVE.md",
    ):
        dest.write_text(text)


def write_patch_analysis_md(fig_dir):
    text = """# Patch retrieval — how to analyze

## Checklist

1. Open `baseline_patch_retrieve/{case}/retrieval_log.json`
2. Inspect top-3 `patch_*_20x.jpg` per node — match node question?
3. Compare path graph: GT vs pred divergence point
4. Paired metrics: Patch vs A on same slide_id

## Good signs

- Patches match node morphology (glands/endometrium, spindle/myometrium, etc.)
- Spatially diverse coords (d_min filter)
- Rank order consistent; absolute sim ~0.03–0.05 is normal on morphologic nodes

## Similarity scores (cosine)

- `sim` = cosine between TITAN text query embedding and patch image embedding.
- **Positive small values (~0.03–0.05)** on `endometrium_assessment` / `mass_histologic_type` = expected.
- **Negative values (~−0.01 to −0.03)** on `synthesis_interpretation` / `diagnosis` = normal:
  abstract text queries do not align well with H&E patches in embedding space.
- Judge retrieval by **rank order and patch morphology**, not absolute sim sign.

## Bad signs

- Artifact/glass tiles; all patches from one corner
- Wrong compartment at root — patches cannot fix routing
- Missing retrieval_log (no embeddings)

See `wrong_root_still_improves_panel.png` for cases where local evidence helps despite wrong root.
"""
    for dest in (
        fig_dir / "PATCH_RETRIEVAL_ANALYSIS.md",
        REPO / "notebooks" / "PATCH_RETRIEVAL_ANALYSIS.md",
    ):
        dest.write_text(text)


def generate_all_figures(repo=None, fig_run_id=None):
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    repo = repo or REPO
    cfg = yaml.safe_load((repo / "configs" / "paths.yaml").read_text())
    runs_dir = Path(cfg["user"]["work_dir"]) / "runs"
    chains_path = repo / "data" / "labels" / "chains.jsonl"
    cache_root = Path(cfg["user"]["cache_dir"])
    fig_run_id = fig_run_id or run_timestamp()
    fig_dir = repo / "notebooks" / "figures" / fig_run_id
    flat_dir = repo / "notebooks" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = resolve_predictions(runs_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No predictions in {runs_dir}")

    gt_test = {
        k: v for k, v in load_jsonl(chains_path).items() if v.get("split") == "test"
    }
    preds_map = {label: load_jsonl(p) for label, p in pred_paths.items()}

    rows = []
    for label, preds in preds_map.items():
        for sid in sorted(set(preds) & set(gt_test)):
            m = case_metrics(preds[sid], gt_test[sid])
            m.update({"slide_id": sid, "baseline": label})
            rows.append(m)
    df = pd.DataFrame(rows)

    mean_tbl = df.groupby("baseline")[CHAIN_METRICS + REPORT_METRICS].mean().round(4)
    mean_tbl["n"] = df.groupby("baseline").size()
    mean_tbl.to_csv(fig_dir / "metrics_summary.csv")
    mean_tbl.to_csv(flat_dir / "metrics_summary.csv")

    plot_graph_overview(fig_dir, flat_dir)
    plot_chain_metrics_grouped(df, fig_dir, flat_dir)
    plot_node_confusion_paired(preds_map, gt_test, fig_dir, flat_dir)
    plot_compartment_stacked(preds_map, gt_test, fig_dir, flat_dir)
    plot_paths_best_worst_grid(df, gt_test, preds_map, fig_dir, flat_dir)
    plot_node_heatmap_all_baselines(preds_map, gt_test, fig_dir, flat_dir)
    plot_edge_f1_scatter(df, fig_dir, flat_dir)
    if "A (flat thumbnail)" in preds_map:
        plot_organ_procedure_confusion_all_baselines(preds_map, gt_test, fig_dir, flat_dir)
    plot_first_divergence(df, fig_dir, flat_dir)
    plot_report_metrics(df, fig_dir, flat_dir)
    plot_report_length(df, fig_dir, flat_dir)
    plot_report_text_highlights(df, gt_test, preds_map, fig_dir, flat_dir)
    plot_chain_mess_highlights(df, gt_test, preds_map, fig_dir, flat_dir)
    plot_rag_effect_summary(df, preds_map, gt_test, fig_dir, flat_dir)
    plot_wrong_root_still_improves_panel(
        find_wrong_root_improved_cases(df, gt_test, preds_map, runs_dir=runs_dir),
        gt_test,
        preds_map,
        df,
        runs_dir,
        cache_root,
        fig_dir,
        flat_dir,
    )
    if "A (flat thumbnail)" in preds_map and "Patch (k=100 centroids)" in preds_map:
        plot_paired_patch_vs_a(
            df,
            preds_map["A (flat thumbnail)"],
            preds_map["Patch (k=100 centroids)"],
            fig_dir,
            flat_dir,
        )
        patch_df = df[df["baseline"] == "Patch (k=100 centroids)"].sort_values("edge_f1")
        if not patch_df.empty:
            plot_patch_retrieval_manifest(
                patch_df.iloc[-1]["slide_id"],
                runs_dir,
                "best",
                fig_dir,
                flat_dir,
                "patch_retrieval_manifest_best.png",
            )
            plot_patch_retrieval_manifest(
                patch_df.iloc[0]["slide_id"],
                runs_dir,
                "worst",
                fig_dir,
                flat_dir,
                "patch_retrieval_manifest_worst.png",
            )
            plot_example_path_graph_single(
                patch_df.iloc[-1]["slide_id"],
                gt_test,
                preds_map,
                df,
                fig_dir,
                flat_dir,
                "patch_best",
            )
    plot_patch_coverage(cache_root, chains_path, fig_dir, flat_dir)
    write_narrative_md(fig_dir, mean_tbl)
    write_patch_analysis_md(fig_dir)
    return fig_dir
