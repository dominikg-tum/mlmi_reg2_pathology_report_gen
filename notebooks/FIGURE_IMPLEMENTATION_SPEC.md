# Figure implementation spec (v2) — ready to execute in Agent mode

Run after switching to **Agent mode**:

```bash
python notebooks/generate_presentation_figures.py
jupyter nbconvert --to notebook --execute notebooks/baseline_eval_analysis.ipynb \
  --output notebooks/baseline_eval_analysis_executed.ipynb
```

## Files to create/update

1. **`notebooks/figure_utils.py`** (new) — all plot functions
2. **`notebooks/generate_presentation_figures.py`** — thin wrapper: `from figure_utils import generate_all_figures`
3. **`notebooks/baseline_eval_analysis.ipynb`** — remove clinical proxy; `%run` or import figure_utils
4. **`notebooks/PATCH_RETRIEVAL_ANALYSIS.md`** — analysis checklist

## Remove

- `clinical_proxy` from all metrics/plots
- `example_path_comparison.png` (table)
- `example_b1_collapse_flow.png` (B1-failure focus)

## New figures

| File | Description |
|------|-------------|
| `example_paths_best_worst_grid.png` | 4 baselines × best/worst Edge-F1; green/red node boxes along GT path |
| `example_path_graph_*.png` | Single case: GT + A/B1/B2/Patch rows |
| `node_confusion_paired_<node>.png` | GT×Pred heatmap per baseline, same cases |
| `node_accuracy_paired_summary.png` | Grouped bar accuracy by node |
| `wrong_root_still_improves_panel.png` | Thumbnail \| B2 HybridRAG text \| 20× patch mosaic |
| `report_text_highlights.png` | Top-3 ROUGE/BLEU cases, side-by-side text |
| `chain_mess_highlights.png` | Top-2 chain MESS — CoT answers not report |
| `patch_retrieval_manifest_best.png` | Table + patch mosaic from retrieval_log.json |
| `patch_retrieval_manifest_worst.png` | Same for worst patch case |

## Wrong-root case selection

```python
wrong_root = organ_procedure or compartment != GT (for baseline A)
improved = edge_f1(B2 or Patch) > edge_f1(A) + 0.05
pick top 2 by delta
```

## B2 text panel

Recompute via `HybridRAGMemory.retrieve()` at `compartment` node (not stored in cot_chain).

## Patch manifest

Read `dominik/runs/baseline_patch_retrieve/{slide_id}/retrieval_log.json` — fields: node_id, zoom, index, coord, similarity, patch_path JPEGs.

## Slide order (14)

See `PRESENTATION_NARRATIVE.md` after regeneration.
