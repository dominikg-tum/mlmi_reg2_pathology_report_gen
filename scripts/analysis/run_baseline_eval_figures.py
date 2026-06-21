"""Generate baseline eval figures for notebooks/baseline_eval_analysis.ipynb."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.case_slides import iter_chain_records, primary_wsi_for_baseline
from eval.edge_parser import chain_dict_to_record
from eval.metrics.chain import binary_path_validity, edge_f1, mess_score
from eval.metrics.report import bleu4, clinical_accuracy_placeholder, rouge_l
from scripts.inference.run_test_baseline_batch import _has_patch_embeddings

CHAIN_METRICS = ["bpv", "edge_f1", "mess"]
REPORT_METRICS = ["rouge_l", "bleu4", "clinical_proxy"]
METRIC_LABELS = {
    "bpv": "Binary Path Validity",
    "edge_f1": "Edge-F1",
    "mess": "MESS",
    "rouge_l": "ROUGE-L",
    "bleu4": "BLEU-4",
    "clinical_proxy": "Clinical (token F1)",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
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


def first_divergence(pred: dict, gt: dict) -> str | None:
    pa, ga = node_answers(pred), node_answers(gt)
    for nid in gt.get("node_path") or list(ga.keys()):
        if nid in pa and pa[nid] != ga.get(nid):
            return nid
    return None


def case_metrics(pred_raw: dict, gt_raw: dict) -> dict:
    p = chain_dict_to_record(pred_raw)
    g = chain_dict_to_record(gt_raw)
    f1 = edge_f1(p, g)
    return {
        "bpv": binary_path_validity(p, g),
        "edge_f1": f1["f1"],
        "mess": mess_score(p, g),
        "rouge_l": rouge_l(p.report, g.report),
        "bleu4": bleu4(p.report, g.report),
        "clinical_proxy": clinical_accuracy_placeholder(p.report, g.report),
        "pred_report_len": len(p.report or ""),
        "gt_report_len": len(g.report or ""),
        "first_div_node": first_divergence(pred_raw, gt_raw),
    }


def compartment_counts(preds: dict[str, dict]) -> Counter:
    counts: Counter = Counter()
    for rec in preds.values():
        for step in rec.get("chain-of-thought", []):
            if step.get("node_id") == "compartment":
                counts[step.get("answer", "?")] += 1
    return counts


def plot_metric_bars(data: pd.DataFrame, metrics: list[str], title: str, path: Path, palette) -> None:
    plot_df = data.groupby("baseline")[metrics].mean().reset_index()
    long = plot_df.melt(id_vars="baseline", var_name="metric", value_name="score")
    long["metric"] = long["metric"].map(METRIC_LABELS)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=long, x="metric", y="score", hue="baseline", palette=palette, ax=ax)
    ax.set_ylim(0, max(0.5, long["score"].max() * 1.15))
    ax.set_title(title)
    ax.set_ylabel("Score (mean)")
    ax.legend(title="Baseline", bbox_to_anchor=(1.02, 1), loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/paths.yaml").read_text())
    runs_dir = Path(cfg["user"]["work_dir"]) / "runs"
    chains_path = REPO / "data" / "labels" / "chains.jsonl"
    cache_root = Path(cfg["user"]["cache_dir"])
    baselines = {
        "A (flat thumbnail)": runs_dir / "predictions_test_baseline_a.jsonl",
        "B1 (HippoRAG2)": runs_dir / "predictions_test_baseline_b1.jsonl",
        "B2 (HybridRAG)": runs_dir / "predictions_test_baseline_b2.jsonl",
        "Patch (k=100 centroids)": runs_dir / "predictions_test_baseline_patch_retrieve.jsonl",
    }
    fig_dir = REPO / "notebooks" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = sns.color_palette("Set2", len(baselines) + 1)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

    gt_test = {k: v for k, v in load_jsonl(chains_path).items() if v.get("split") == "test"}
    rows = []
    for name, path in baselines.items():
        if not path.exists():
            continue
        preds = load_jsonl(path)
        for sid in sorted(set(preds) & set(gt_test)):
            m = case_metrics(preds[sid], gt_test[sid])
            m.update(slide_id=sid, baseline=name)
            rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No prediction files found — run baselines first.")

    plot_metric_bars(df, CHAIN_METRICS, "Chain metrics — test split", fig_dir / "chain_metrics_comparison.png", palette)
    plot_metric_bars(df, REPORT_METRICS, "Report metrics", fig_dir / "report_metrics_comparison.png", palette)

    comp_rows = []
    for ans, cnt in compartment_counts(gt_test).items():
        comp_rows.append({"baseline": "GT (test)", "compartment": ans, "count": cnt})
    for label, path in baselines.items():
        if path.exists():
            for ans, cnt in compartment_counts(load_jsonl(path)).items():
                comp_rows.append({"baseline": label, "compartment": ans, "count": cnt})
    comp_df = pd.DataFrame(comp_rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    order = comp_df.groupby("compartment")["count"].sum().sort_values(ascending=False).index
    sns.barplot(data=comp_df, x="compartment", y="count", hue="baseline", order=order, ax=ax)
    ax.set_title("Predicted compartment distribution vs GT")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Source")
    plt.tight_layout()
    fig.savefig(fig_dir / "compartment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    a_df = df[df["baseline"] == "A (flat thumbnail)"]
    if not a_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        a_df["first_div_node"].value_counts().head(10).sort_values().plot(kind="barh", ax=ax, color=palette[0])
        ax.set_title("Baseline A — first divergence node")
        plt.tight_layout()
        fig.savefig(fig_dir / "first_divergence_node_A.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    a_path = baselines["A (flat thumbnail)"]
    if a_path.exists():
        a_preds = load_jsonl(a_path)
        pairs = []
        for sid in a_preds:
            if sid not in gt_test:
                continue
            ga = node_answers(gt_test[sid]).get("organ_procedure")
            pa = node_answers(a_preds[sid]).get("organ_procedure")
            if ga and pa:
                pairs.append((ga, pa))
        if pairs:
            ct = pd.crosstab(pd.Series([p[0] for p in pairs]), pd.Series([p[1] for p in pairs]))
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("organ_procedure confusion — Baseline A")
            plt.tight_layout()
            fig.savefig(fig_dir / "organ_procedure_confusion_A.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        key_nodes = ["organ_procedure", "compartment", "endometrium_assessment", "diagnosis", "mass_histologic_type"]
        acc_rows = []
        for nid in key_nodes:
            correct = total = 0
            for sid in a_preds:
                if sid not in gt_test:
                    continue
                ga = node_answers(gt_test[sid]).get(nid)
                pa = node_answers(a_preds[sid]).get(nid)
                if ga is None:
                    continue
                total += 1
                if pa == ga:
                    correct += 1
            acc_rows.append({"node": nid, "accuracy": correct / total if total else 0, "n": total})
        acc_df = pd.DataFrame(acc_rows).set_index("node")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            acc_df[["accuracy"]].T,
            annot=acc_df["n"].values.reshape(1, -1),
            fmt="d",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Node-level accuracy — Baseline A")
        plt.tight_layout()
        fig.savefig(fig_dir / "node_accuracy_heatmap_A.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    len_rows = []
    for _, row in df.iterrows():
        len_rows.append({"baseline": row["baseline"], "length": row["pred_report_len"], "type": "Prediction"})
    for _, row in df.drop_duplicates("slide_id").iterrows():
        len_rows.append({"baseline": "GT", "length": row["gt_report_len"], "type": "Ground truth"})
    len_df = pd.DataFrame(len_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=len_df, x="type", y="length", hue="baseline", ax=ax)
    ax.set_title("Report length — pred vs GT")
    plt.tight_layout()
    fig.savefig(fig_dir / "report_length_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    p_path = baselines["Patch (k=100 centroids)"]
    if a_path.exists() and p_path.exists():
        shared = sorted(set(load_jsonl(a_path)) & set(load_jsonl(p_path)))
        fair = df[df["slide_id"].isin(shared) & df["baseline"].isin(["A (flat thumbnail)", "Patch (k=100 centroids)"])]
        pivot = fair.pivot(index="slide_id", columns="baseline", values="edge_f1").dropna()
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(pivot["A (flat thumbnail)"], pivot["Patch (k=100 centroids)"], alpha=0.7, s=60)
            lim = [0, max(pivot.max()) * 1.05 + 0.01]
            ax.plot(lim, lim, "k--", alpha=0.4)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("Edge-F1 — A")
            ax.set_ylabel("Edge-F1 — Patch")
            ax.set_title(f"Paired Edge-F1 ({len(pivot)} cases)")
            plt.tight_layout()
            fig.savefig(fig_dir / "paired_edge_f1_patch_vs_A.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x="baseline", y="edge_f1", hue="baseline", inner="quartile", palette=palette, ax=ax, legend=False)
    ax.set_title("Edge-F1 per case")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(fig_dir / "edge_f1_violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cov = [{"has_emb": _has_patch_embeddings(cache_root, primary_wsi_for_baseline(rec["slide_id"]))}
           for rec in iter_chain_records(chains_path, split="test")]
    cov_df = pd.DataFrame(cov)
    fig, ax = plt.subplots(figsize=(5, 4))
    cov_df["has_emb"].value_counts().rename({True: "encoded", False: "missing"}).plot(
        kind="bar", ax=ax, color=[palette[2], palette[1]]
    )
    ax.set_title("Patch embedding coverage")
    plt.tight_layout()
    fig.savefig(fig_dir / "patch_embedding_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {len(list(fig_dir.glob('*.png')))} figures -> {fig_dir}")
    print(df.groupby("baseline")[CHAIN_METRICS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
