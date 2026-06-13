# WSI Background / Tissue Filtering

*For: Agentic Pathology Report Generation (TUM Untera uterine H&E, ~220 slides)*

**See also:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (Phase 0 offline preprocessing), [WSI_Patching_Retrieval_Research.md](WSI_Patching_Retrieval_Research.md) (patch pools & retrieval).

---

## 1. Why this matters

Background filtering runs **offline only** — during tiling before CONCH encode. It decides which patch coordinates enter `coords_{zoom}.pt` and therefore:

- how many patches get encoded at each of the four zoom levels
- K-means centroid quality (glass-heavy centroids hurt cosine retrieval)
- TITAN slide embedding input (×20 pool only)

It does **not** run at agent inference time. Retrieval and CMT adjacent-scale enrichment operate on whatever was pre-filtered.

---

## 2. Current implementation (shipped)

| Item | Detail |
|------|--------|
| Location | `vision/wsi_io.py` → `is_tissue_patch()`; called from `iter_tissue_patches()` |
| Rule | Keep patch if **grayscale mean ≤ 220** (RGB averaged per pixel, then mean over patch) |
| Config | **None** — threshold hardcoded; not in `configs/vision.yaml` |
| Docs | Previously undocumented |
| Scope | All four zoom levels use the same per-patch check during tiling |

```python
# vision/wsi_io.py (current)
def is_tissue_patch(arr, background_threshold=220) -> bool:
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    return float(gray.mean()) <= background_threshold
```

**Strengths:** trivial, fast, no extra dependencies, works on pure-glass tiles.

**Weaknesses:**

| Failure mode | Example |
|--------------|---------|
| Edge patches (half glass, half tissue) | Mean sits mid-range → often **kept** |
| Mostly-glass + small tissue speck | Dark speck pulls mean down → **kept** |
| Pale stroma / fat / necrosis | Bright but diagnostic → sometimes **dropped** |
| Stain / scanner variation | Fixed 220 does not adapt per slide |

---

## 3. Candidate methods (comparison)

| # | Method | Where applied | Adaptive? | Edge-patch handling | Cost | Verdict |
|---|--------|---------------|-----------|---------------------|------|---------|
| A | **Mean luminance threshold** (current) | Per patch | No | Poor | Minimal | Bootstrap only |
| B | **Dark-pixel fraction** `(gray < T).mean() ≥ f_min` | Per patch | No | Better than A | Minimal | Quick incremental fix |
| C | **Otsu on each patch** | Per patch | Per patch | Unstable on partial tissue | High (×4 zooms × all tiles) | **Not recommended** |
| D | **Otsu on slide thumbnail** → binary mask | Once per slide | Per slide | Good (with fraction check) | Low | Good mask backend |
| E | **HSV saturation + value mask** on thumbnail | Once per slide | Tunable globals | Good | Low | **Best mask backend** |
| F | **Lab distance from glass colour** | Once per slide | Tunable | Good on H&E | Low–medium | Alternative to E |
| G | **Deep tissue segmentation** (U-Net etc.) | Once per slide | Learned | Best | GPU + model maintenance | Overkill for 220 slides |

### Why not per-patch Otsu?

On a 512×512 tile that is 90% glass, Otsu often splits noise/artifacts rather than tissue vs background. Running it at four zoom levels multiplies CPU cost with little gain over a **single slide-level mask**.

### Why a slide-level mask?

- One mask per WSI, reused for 5× / 10× / 20× / 40× tiling (same level-0 coordinate grid)
- Thumbnails already exist on cluster — `dataset/thumbnails/`, `thumbnails_kmeans/`, `thumbnails_kmeans_5/` (see [PROJECT_OVERVIEW.md §2a](PROJECT_OVERVIEW.md#2a-thumbnail-options-cluster))
- Matches common WSI practice (CLAM-style pipelines, MahmoodLab ecosystem expectations)
- Easy to QA in `verify_tiling.py` (mask overlay + patch count before CONCH)

---

## 4. Recommended approach (most robust for this project)

### **Slide-level tissue mask + minimum tissue fraction per patch**

This is the best balance of **robustness**, **cost**, and **fit with the existing pipeline**.

#### Pipeline

```text
Per slide (offline, once):
  1. Load thumbnail (max edge 1024) or openslide low-res level
  2. Build binary tissue mask (HSV — see below)
  3. Optional: morphological close/open to remove speckle & fill holes
  4. Save tissue_mask.png + meta (level-0 scale factor)

Per candidate patch (all zoom levels):
  1. Map patch bbox to mask coordinates
  2. tissue_fraction = (mask pixels inside bbox) / (patch area)
  3. Keep iff tissue_fraction >= min_tissue_fraction
```

#### Mask construction (recommended: HSV)

H&E tissue is pink/purple (saturation) vs white/clear glass (low saturation, high value):

```text
tissue pixel  :=  saturation > sat_min  AND  value < val_max
```

Suggested starting values for uterine H&E:

| Parameter | Start value | Notes |
|-----------|-------------|-------|
| `sat_min` | 0.08 (HSV S channel, 0–1) | Raise if glass leaks in |
| `val_max` | 0.95 (HSV V channel, 0–1) | Lower if bright fold marks pass |
| `min_tissue_fraction` | **0.40** | Fraction of patch area that must be tissue |
| Morph close kernel | 5×5 | Optional; removes pinholes in tissue |

If HSV leaves large glass islands on a subset of slides, add **Otsu-on-thumbnail** as a second mask method behind config — not as the default.

#### Why this beats the alternatives

| Criterion | Mean ≤ 220 | Per-patch Otsu | **Mask + fraction** |
|-----------|------------|----------------|---------------------|
| Handles edge patches | ❌ | ⚠️ | ✅ |
| Adapts per slide | ❌ | ⚠️ | ✅ (mask) |
| Single compute per slide | ✅ | ❌ | ✅ |
| Works across 4 zoom levels | ✅ | ✅ | ✅ (shared mask) |
| Visual QA | Hard | Hard | Easy overlay |
| Dependencies | None | skimage | numpy + PIL (HSV via PIL/RGB) |

#### Proposed config (not yet wired)

Add to `configs/vision.yaml`:

```yaml
tissue_filter:
  method: slide_mask          # slide_mask | mean_threshold | dark_fraction
  mask_source: thumbnail      # thumbnail | openslide_lowres
  min_tissue_fraction: 0.40
  hsv_sat_min: 0.08
  hsv_val_max: 0.95
  morph_close_px: 5           # 0 = disable
  # Legacy fallback (method: mean_threshold)
  background_threshold: 220
  # Quick win (method: dark_fraction)
  dark_fraction_min: 0.25
```

Keep `mean_threshold` as legacy until re-tiling is validated — existing caches stay reproducible.

---

## 5. Planned code touchpoints (when implemented)

| File | Change |
|------|--------|
| `vision/tissue_mask.py` | **New** — build mask from thumbnail, scale to level-0, overlap query |
| `vision/wsi_io.py` | `iter_tissue_patches()` — call mask overlap or legacy threshold from config |
| `vision/mag_config.py` | `tissue_filter_config()` helper |
| `configs/vision.yaml` | `tissue_filter` block above |
| `scripts/vision/verify_tiling.py` | Mask overlay montage + patch count vs legacy |
| `scripts/vision/tile_slides.py` | Save `tissue_mask.png` per slide cache dir |

No changes to retrieval, agent inference, or CMT adjacent-scale logic.

---

## 6. Validation plan (before re-tiling full corpus)

Run on **~10 representative slides** (small biopsy, large hysterectomy, fatty, necrotic, ink/pen if any):

1. **Patch counts** per zoom vs current mean-220 baseline — expect fewer tiles on glass-heavy margins, stable counts on tissue-dense regions.
2. **Montage** — `verify_tiling.py` spread sample; confirm no obvious glass tiles.
3. **Mask overlay** — visual check: endometrium/myometrium covered, glass excluded.
4. **Downstream spot-check** — re-encode one slide, run `run_retrieval_demo.py`; retrieval should not regress.

Tune `min_tissue_fraction` down to **0.30** if pale compartments are over-filtered; up to **0.50** if glass edge patches persist.

---

## 7. Decision summary

| Question | Answer |
|----------|--------|
| Best robust method for this project? | **Slide-level HSV tissue mask + `min_tissue_fraction` per patch** |
| Use Otsu? | **Once on thumbnail**, optional fallback — not per patch |
| Use deep segmentation? | **No** unless mask QA fails on many slides |
| Keep current mean-220? | **Legacy fallback only** until re-tile validated |
| Quick win without mask plumbing? | **Dark-pixel fraction** — better edges, still not slide-adaptive |

---

## 8. References & lineage

| Resource | Relevance |
|----------|-----------|
| CLAM / WSI patching literature | Slide-level tissue detection before patch extraction |
| MahmoodLab TITAN / CONCH | Expect reasonable tissue-only pools upstream |
| Current code | `vision/wsi_io.py`, `vision/patching.py`, `scripts/vision/tile_slides.py` |
