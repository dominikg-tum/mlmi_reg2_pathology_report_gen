# Presentation narrative — REG² baseline eval (v2)

## Recommended slide order (14 slides)

| # | Slide | Figure file |
|---|-------|-------------|
| 1 | **Problem** — Agent walks a fixed diagnostic graph | `graph_overview_simplified.png` |
| 2 | **Full graph** (backup) — 20 nodes | `graph_overview_full.png` |
| 3 | **Setup** — 69 test cases, GT=text-oracle, pred=vision on 1 WSI | (bullet slide) |
| 4 | **Chain metrics** — BPV, Edge-F1, MESS | `chain_metrics_comparison.png` |
| 5 | **Where A fails** — organ_procedure + first divergence | `organ_procedure_confusion_A.png`, `organ_procedure_confusion_all_baselines.png`, `first_divergence_node_A.png` |
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

| baseline                |    bpv |   edge_f1 |   mess |   rouge_l |   bleu4 |   n |
|:------------------------|-------:|----------:|-------:|----------:|--------:|----:|
| A (flat thumbnail)      | 0.1739 |    0.3121 | 0.3145 |    0.1364 |  0.1364 |  69 |
| B1 (HippoRAG2)          | 0.1739 |    0.3207 | 0.3243 |    0.1379 |  0.1379 |  69 |
| B2 (HybridRAG)          | 0.1739 |    0.3972 | 0.4136 |    0.177  |  0.177  |  69 |
| Patch (k=100 centroids) | 0.1364 |    0.2824 | 0.3178 |    0.0259 |  0.0259 |  22 |
