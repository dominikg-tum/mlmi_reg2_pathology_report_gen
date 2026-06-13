"""Render data/graph/execution_graph.jsonl as a labelled flow diagram.

Pure-matplotlib (no graphviz/networkx). Reads the JSONL so the picture always
matches the live graph. Draws every node (id, zoom, visual policy, interaction)
and every edge (labelled with the answer that triggers it).

Readability strategy:
  * wide column spacing so every edge label lives in a clear vertical "channel"
    immediately to the right of its source node;
  * each source node's outgoing edges share a colour, so crossing edges can be
    traced back to where they came from;
  * a collision-avoidance pass nudges labels vertically so none overlap.

Usage:
    python scripts/visualize_uterus_graph.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "execution_graph.jsonl"
OUT_PNG = REPO_ROOT / "docs" / "uterus_execution_graph.png"

KIND_COLOR = {
    "global": "#AED6F1",
    "compartment": "#A9DFBF",
    "local": "#F9E79F",
    "integration": "#F5CBA7",
    "report": "#F1948A",
}

# Manual (col, row) layout — left-to-right diagnostic flow, family-grouped rows.
POS = {
    "organ_procedure": (0, 1.0),
    "compartment": (1, 1.0),
    # endometrium family (top)
    "endometrium_assessment": (2, 8.5),
    "endometrium_cycle_phase": (3, 11.0),
    "endometritis_type": (3, 9.3),
    "endometrial_hyperplasia_grade": (3, 7.6),
    "endometrial_carcinoma_subtype": (3, 5.6),
    "endometrial_carcinoma_grade": (4, 5.6),
    "background_endometrium": (5, 5.6),
    # myometrium family
    "myometrium_assessment": (2, 3.0),
    "smooth_muscle_tumor_assessment": (3, 3.0),
    # junctional zone / serosa
    "junctional_zone_assessment": (2, 1.0),
    "serosa_assessment": (2, -0.8),
    # mass family (bottom)
    "mass_histologic_type": (2, -3.2),
    "microscopic_pattern": (3, -3.2),
    "cellular_features": (4, -3.2),
    # shared integration tail (right, centred)
    "stage_extent": (6, 1.0),
    "synthesis_interpretation": (7, 1.0),
    "diagnosis": (8, 1.0),
    "report": (9, 1.0),
}

COL_W = 5.4
ROW_H = 1.5
BOX_W = 1.45  # half-width in data coords
BOX_H = 0.62  # half-height in data coords
GAP = COL_W - 2 * BOX_W  # horizontal room between adjacent column node edges


def load_nodes() -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    with GRAPH_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                nodes[obj["id"]] = obj
    return nodes


def xy(node_id: str) -> tuple[float, float]:
    col, row = POS[node_id]
    return col * COL_W, row * ROW_H


def deoverlap(labels: list[dict], *, min_gap: float = 0.46, bin_w: float = 1.4) -> None:
    """Push labels apart vertically within each x-channel so none overlap."""
    bins: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        bins[round(lab["x"] / bin_w)].append(i)
    for idxs in bins.values():
        idxs.sort(key=lambda i: labels[i]["y"])
        for j in range(1, len(idxs)):
            prev_y = labels[idxs[j - 1]]["y"]
            if labels[idxs[j]]["y"] - prev_y < min_gap:
                labels[idxs[j]]["y"] = prev_y + min_gap


def draw() -> None:
    nodes = load_nodes()
    missing = set(nodes) - set(POS)
    if missing:
        raise SystemExit(f"No layout position for nodes: {sorted(missing)}")

    # stable colour per source node (for edge tracing)
    cmap = plt.get_cmap("tab20")
    edge_color = {nid: cmap(i % 20) for i, nid in enumerate(sorted(nodes))}

    fig, ax = plt.subplots(figsize=(40, 22))

    label_entries: list[dict] = []

    # ---- edges (arrows now; labels collected, de-overlapped, drawn after) ----
    for nid, node in nodes.items():
        sx, sy = xy(nid)
        ecolor = edge_color[nid]
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
            arrow = FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.3,
                color=ecolor,
                linestyle=(0, (4, 3)) if is_default else "solid",
                alpha=0.45 if long_edge else 0.85,
                zorder=1,
            )
            ax.add_patch(arrow)

            # label sits in the channel immediately right of the source node
            label_x = sx + BOX_W + GAP * 0.5
            denom = (end[0] - start[0]) or 1e-6
            t = (label_x - start[0]) / denom
            t = min(max(t, 0.0), 1.0)
            label_y = start[1] + (end[1] - start[1]) * t + rad * 1.4
            label_entries.append(
                {
                    "x": label_x,
                    "y": label_y,
                    "text": "any answer (multi-select)" if is_default else answer,
                    "color": ecolor,
                    "is_default": is_default,
                }
            )

    deoverlap(label_entries)

    for lab in label_entries:
        ax.text(
            lab["x"],
            lab["y"],
            lab["text"],
            fontsize=7.2,
            color="#7b241c" if lab["is_default"] else "#17202a",
            ha="center",
            va="center",
            zorder=2,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec=lab["color"],
                lw=1.0,
                alpha=0.95,
            ),
        )

    # ---- nodes (on top) ----
    for nid, node in nodes.items():
        cx, cy = xy(nid)
        color = KIND_COLOR.get(node["node_kind"], "#D5D8DC")
        box = FancyBboxPatch(
            (cx - BOX_W, cy - BOX_H),
            2 * BOX_W,
            2 * BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=color,
            ec="#1b2631",
            lw=2.0 if (node.get("root") or node.get("is_leaf")) else 0.9,
            zorder=3,
        )
        ax.add_patch(box)

        tag = "  [ROOT]" if node.get("root") else ("  [LEAF]" if node.get("is_leaf") else "")
        ax.text(cx, cy + 0.34, node["label"] + tag, fontsize=9.2, fontweight="bold",
                ha="center", va="center", zorder=4, color="#0b0b0b")
        ax.text(cx, cy + 0.07, nid, fontsize=7.0, ha="center", va="center",
                style="italic", color="#566573", zorder=4)
        ax.text(cx, cy - 0.18, f"{node['zoom_level']} | {node['visual_policy']}",
                fontsize=6.6, ha="center", va="center", color="#212f3d", zorder=4)
        ax.text(cx, cy - 0.40, node["interaction"], fontsize=6.4, ha="center",
                va="center", color="#7d6608", zorder=4)

    # ---- legend ----
    handles = [mpatches.Patch(fc=c, ec="#1b2631", label=k) for k, c in KIND_COLOR.items()]
    handles.append(mpatches.Patch(fc="white", ec="#7b241c", label="dashed = multi-select __default__"))
    handles.append(mpatches.Patch(fc="white", ec="#888888", label="edge colour = source node"))
    ax.legend(handles=handles, title="node_kind", loc="lower left", fontsize=11,
              title_fontsize=12, framealpha=0.96).set_zorder(5)

    xs = [xy(n)[0] for n in nodes]
    ys = [xy(n)[1] for n in nodes]
    ax.set_xlim(min(xs) - 2.4, max(xs) + 2.8)
    ax.set_ylim(min(ys) - 3.0, max(ys) + 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Uterus diagnostic execution graph  (data/graph/execution_graph.jsonl)  -  "
        f"{len(nodes)} nodes",
        fontsize=17, fontweight="bold", pad=18,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    draw()
