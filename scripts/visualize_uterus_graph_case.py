"""Highlight a canonical case path on the uterus execution graph.

Reuses layout/colors from visualize_uterus_graph.py. Questions are drawn next
to path nodes (plain text, no box). Answers stay on the path edges (boxed).

Usage:
    python scripts/visualize_uterus_graph_case.py
    python scripts/visualize_uterus_graph_case.py --case endometrioid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from visualize_uterus_graph import (  # noqa: E402  # same scripts/ dir on sys.path
    BOX_H,
    BOX_W,
    GAP,
    KIND_COLOR,
    POS,
    load_nodes,
    xy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CASE_PNG = REPO_ROOT / "docs" / "uterus_execution_graph_case_endometrioid.png"
HI = "#C0392B"

# Canonical video walkthrough: endometrioid adenocarcinoma, FIGO grade 2.
# Each tuple is (source_node_id, answer_key used on that edge).
CASE_ENDOMETRIOID: list[tuple[str, str]] = [
    ("organ_procedure", "uterus_hysterectomy"),
    ("compartment", "endometrium"),
    ("endometrium_assessment", "carcinoma"),
    ("endometrial_carcinoma_subtype", "endometrioid"),
    ("endometrial_carcinoma_grade", "grade_2"),
    ("background_endometrium", "hyperplastic_background"),
    ("stage_extent", "superficial_invasion"),
    ("synthesis_interpretation", "definitive"),
    ("diagnosis", "malignant"),
]
CASES = {"endometrioid": CASE_ENDOMETRIOID}


def wrap_question(text: str, width: int = 28) -> str:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and len(trial) > width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def draw(case: str = "endometrioid") -> None:
    nodes = load_nodes()
    missing = set(nodes) - set(POS)
    if missing:
        raise SystemExit(f"No layout position for nodes: {sorted(missing)}")

    path = CASES[case]
    highlight_edges: set[tuple[str, str]] = set()
    highlight_nodes: set[str] = set()
    step_num: dict[str, int] = {}
    for i, (src, answer) in enumerate(path, start=1):
        if answer not in nodes[src]["edges"]:
            raise SystemExit(f"Case edge missing: {src} --{answer}-->")
        highlight_edges.add((src, answer))
        highlight_nodes.add(src)
        step_num[src] = i
        highlight_nodes.add(nodes[src]["edges"][answer])
    step_num["report"] = len(path) + 1

    fig, ax = plt.subplots(figsize=(40, 24))
    answer_labels: list[dict] = []

    # ---- edges ----
    for nid, node in nodes.items():
        sx, sy = xy(nid)
        items = list(node.get("edges", {}).items())
        n_out = len(items)
        for i, (answer, target) in enumerate(items):
            tx, ty = xy(target)
            dx_cols = POS[target][0] - POS[nid][0]
            long_edge = dx_cols > 1
            spread = i - (n_out - 1) / 2.0
            if abs(sy - ty) < 0.1 and not long_edge:
                rad = spread * 0.04
            else:
                base = 0.14 if sy >= ty else -0.14
                rad = base + spread * 0.05

            start = (sx + BOX_W, sy)
            end = (tx - BOX_W, ty)
            is_default = answer == "__default__"
            on_path = (nid, answer) in highlight_edges
            if on_path:
                color, lw, alpha, z = HI, 3.2, 0.95, 2
            else:
                color, lw, alpha, z = "#BFC9CA", 0.7, 0.12, 1

            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>",
                    mutation_scale=18 if on_path else 14,
                    lw=lw,
                    color=color,
                    linestyle=(0, (4, 3)) if is_default else "solid",
                    alpha=alpha,
                    zorder=z,
                )
            )

            if not on_path:
                continue

            label_x = sx + BOX_W + GAP * 0.5
            denom = (end[0] - start[0]) or 1e-6
            t = min(max((label_x - start[0]) / denom, 0.0), 1.0)
            label_y = start[1] + (end[1] - start[1]) * t + rad * 1.4
            answer_labels.append(
                {
                    "x": label_x,
                    "y": label_y,
                    "text": "any answer (multi-select)" if is_default else answer,
                }
            )

    # CAP report answer near the leaf (no outgoing edge)
    if "report" in highlight_nodes:
        rx, ry = xy("report")
        answer_labels.append(
            {"x": rx, "y": ry - BOX_H - 0.85, "text": "(CAP report text)"}
        )

    for lab in answer_labels:
        ax.text(
            lab["x"],
            lab["y"],
            lab["text"],
            fontsize=9.0,
            fontweight="bold",
            color="#922B21",
            ha="center",
            va="center",
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="#FDEDEC",
                ec=HI,
                lw=1.8,
                alpha=0.98,
            ),
        )

    # ---- nodes + questions beside path nodes ----
    for nid, node in nodes.items():
        cx, cy = xy(nid)
        on_path = nid in highlight_nodes
        color = KIND_COLOR.get(node["node_kind"], "#D5D8DC") if on_path else "#EAECEE"
        ax.add_patch(
            FancyBboxPatch(
                (cx - BOX_W, cy - BOX_H),
                2 * BOX_W,
                2 * BOX_H,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                fc=color,
                ec=HI if on_path else "#1b2631",
                lw=3.4 if on_path else 0.9,
                zorder=4,
                alpha=1.0 if on_path else 0.35,
            )
        )

        if nid in step_num:
            ax.text(
                cx - BOX_W + 0.18,
                cy + BOX_H - 0.18,
                str(step_num[nid]),
                fontsize=11,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=6,
                bbox=dict(boxstyle="circle,pad=0.22", fc=HI, ec="white", lw=1.2),
            )

        alpha = 1.0 if on_path else 0.35
        tag = "  [ROOT]" if node.get("root") else ("  [LEAF]" if node.get("is_leaf") else "")
        ax.text(cx, cy + 0.34, node["label"] + tag, fontsize=9.2, fontweight="bold",
                ha="center", va="center", zorder=5, color="#0b0b0b", alpha=alpha)
        ax.text(cx, cy + 0.07, nid, fontsize=7.0, ha="center", va="center",
                style="italic", color="#566573", zorder=5, alpha=alpha)
        ax.text(cx, cy - 0.18, f"{node['zoom_level']} | {node['visual_policy']}",
                fontsize=6.6, ha="center", va="center", color="#212f3d", zorder=5, alpha=alpha)
        ax.text(cx, cy - 0.40, node["interaction"], fontsize=6.4, ha="center",
                va="center", color="#7d6608", zorder=5, alpha=alpha)

        # Question next to the node (above), plain text — not with the answer.
        if on_path:
            ax.text(
                cx,
                cy + BOX_H + 0.22,
                wrap_question(node["question"]),
                fontsize=6.8,
                color="#1B2631",
                ha="center",
                va="bottom",
                zorder=5,
                linespacing=1.15,
            )

    handles = [mpatches.Patch(fc=c, ec="#1b2631", label=k) for k, c in KIND_COLOR.items()]
    handles.append(mpatches.Patch(fc="white", ec="#7b241c", label="dashed = multi-select __default__"))
    handles.append(mpatches.Patch(fc="#FDEDEC", ec=HI, label=f"highlighted case: {case}"))
    ax.legend(handles=handles, title="node_kind", loc="lower left", fontsize=11,
              title_fontsize=12, framealpha=0.96).set_zorder(5)

    xs = [xy(n)[0] for n in nodes]
    ys = [xy(n)[1] for n in nodes]
    ax.set_xlim(min(xs) - 2.4, max(xs) + 2.8)
    ax.set_ylim(min(ys) - 3.2, max(ys) + 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Case walkthrough — endometrioid adenocarcinoma, FIGO grade 2  "
        "(follow red numbered boxes 1→10)",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )

    fig.tight_layout()
    fig.savefig(OUT_CASE_PNG, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT_CASE_PNG}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=sorted(CASES), default="endometrioid")
    args = p.parse_args()
    draw(case=args.case)
