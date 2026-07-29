# Patch retrieval — how to analyze

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
