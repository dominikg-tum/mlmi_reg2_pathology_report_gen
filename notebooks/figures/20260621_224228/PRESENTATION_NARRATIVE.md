# Presentation narrative — REG² baseline eval

## Recommended slide order (12 slides)

| # | Slide | Figure file |
|---|-------|-------------|
| 1 | **Problem** — Agent walks a fixed diagnostic graph; eval = chain + report | `graph_overview_simplified.png` |
| 2 | **Full graph** (optional backup) — 20 nodes, deterministic edges | `graph_overview_full.png` |
| 3 | **Setup** — 69 test cases, GT=text-oracle, pred=vision on 1 WSI | (bullet slide) |
| 4 | **Pipeline works** — all baselines complete, auto-eval | metrics table |
| 5 | **Chain metrics** — primary signal (BPV, Edge-F1, MESS) | `chain_metrics_comparison.png` |
| 6 | **Where A fails** — organ_procedure thumbnail-only | `organ_procedure_confusion_A.png` + `first_divergence_node_A.png` |
| 7 | **RAG story** — B1 hurts, B2 helps | `presentation_summary_rag.png` |
| 8 | **B1 collapse detail** — same case, different path | `example_b1_collapse_flow.png` |
| 9 | **Example table** — 3 curated cases | `example_path_comparison.png` |
| 10 | **Node heatmap** — accuracy by graph node | `node_accuracy_heatmap_all_baselines.png` |
| 11 | **Report metrics caveat** — genre mismatch | `report_length_boxplot.png` |
| 12 | **Patch retrieve** — fair n=22 comparison | `paired_patch_vs_A_before_after.png` + `patch_embedding_coverage.png` |

## Which example cases to show live

1. **B1 RAG collapse** (`5e763b37…`) — A: endometrium/physiologic; B1: mass_lesion/malignant. Best demo of RAG overpowering thumbnail.
2. **organ_procedure error** (`31ce2e3a…`) — GT curettage, A predicts hysterectomy at root. Explains 40/69 first-node errors.
3. **Patch improvement** (`1d0a708b…`) — Edge-F1 +0.46 vs thumbnail on encoded case. Shows patch path promise.
4. **Partial success** (`737a8125…`) — Edge-F1 0.73 on A; show the model *can* navigate when thumbnail suffices.

## Key talking points

- **Show the graph** — yes, audience needs to see `organ_procedure` → `compartment` as first two hops with thumbnail-only policy.
- **Do not apologize for low BPV** — explain strict exact-path metric + early-branch errors.
- **Split report vs chain metrics** — ROUGE ~0.03 is genre mismatch, not clinical null result.
- **Never compare patch n=22 to A n=69** — always paired subset.

## Mean metrics (auto-generated)

                            bpv  edge_f1    mess  rouge_l   bleu4   n
baseline                                                             
A (flat thumbnail)       0.1014   0.2680  0.2744   0.0315  0.0315  69
B1 (HippoRAG2)           0.0145   0.1671  0.1584   0.0035  0.0035  69
B2 (HybridRAG)           0.1739   0.3894  0.4083   0.0223  0.0223  69
Patch (k=100 centroids)  0.1364   0.2824  0.3178   0.0259  0.0259  22
