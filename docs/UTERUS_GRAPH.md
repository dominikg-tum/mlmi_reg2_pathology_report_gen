# Uterus Diagnostic Graph — Design & Implementation

**Owner lane:** DOGA · **Artifact:** [`data/graph/execution_graph.jsonl`](../data/graph/execution_graph.jsonl) · **Schema:** [`graph/schema.py`](../graph/schema.py)

This document records how the uterus diagnostic graph was designed and built, how it is
formatted, how it works at inference, how it fits the wider project structure, and what it
contributes to the end-to-end pipeline.

---

## 1. What this artifact is

`data/graph/execution_graph.jsonl` is the **uterus diagnostic graph the agent walks at
inference**. It is a JSONL file — one JSON object (one node) per line. During Phase 1, for
each node:

1. the VLM is given the node's `question` (plus retrieved patches / thumbnail),
2. it produces an answer,
3. the node's `edges` deterministically select the next node from that answer.

**The model never chooses where to go — the graph does.** The graph also tells the system
*how to look* (`zoom_level` selects the patch pool) and *what to show* the VLM
(`visual_policy`: thumbnail, retrieved patches, or both).

### Sources

| Source | Role |
|--------|------|
| `uterus_graph.png` | The original hand-drawn uterus ontology (tiers + medical categories) that this JSONL operationalizes |
| PathoGraph (Sci Data 2025, s41597-025-04906-z) | Clinical *logic* borrowed: staged narrowing, feature → narrowed differential, present-vs-absent branching |
| `docs/PROJECT_OVERVIEW.md` §4–§5 | The authoritative node schema (treated as read-only ground truth) |

---

## 2. Key decisions

| Decision | What we did | Why |
|----------|-------------|-----|
| **Fix `compartment` → `report` shortcut** | `compartment` now routes each option into its own local work-up node, never straight to `report` | The seed graph jumped from compartment selection directly to the report leaf, skipping all diagnostic reasoning. This was the first thing to fix. |
| **Build one full path first** | The **endometrium** branch is the deepest, complete path (assessment → cycle/inflammation/hyperplasia/carcinoma → subtype → grade → background → staging → synthesis → diagnosis → report) | Validates the full tier flow end-to-end before fanning out to other compartments. |
| **Cover all compartments** | Myometrium, junctional zone, serosa/perimetrium, and the mass-lesion → histologic-type branch are all wired and converge on the shared integration tail | Matches every box in `uterus_graph.png`. |
| **Shared neoplastic tail** | `microscopic_pattern` → `cellular_features` → `stage_extent` → `synthesis_interpretation` → `diagnosis` → `report` is reused by every malignant path | Avoids duplicating the integration logic per compartment; keeps the graph small and consistent. |
| **`description` = retrieval text only** | Each `description` says *what to look for on the slide* (structure, cell type, pattern); it is never the displayed question | `description` is concatenated with `question` and encoded by CONCH/TITAN to retrieve patches. Specific morphology → sharper patch selection. |
| **No `40x` anywhere** | Nodes that would ideally be 40x (nuclei, mitoses) use `20x` with sharper descriptions | Only `5x/10x/20x` pools are extracted offline; a `40x` value breaks retrieval. Stable node ids let us flip `zoom_level` later if 40x is added. |
| **Plain edges** | Edges are pure `{answer: next_node_id}` maps — no labels like `suggests_diagnosis` | The traversal engine keys navigation purely on the answer string; semantic edge labels are not part of this schema. |
| **PathoGraph logic, not its labels** | Staged narrowing (preliminary → further → final), present/absent branches (hyperplasia atypia; smooth-muscle malignancy triad), differential convergence at synthesis | Steals the pathologist's reasoning structure without importing PathoGraph's OWL relations. |
| **Updated two coupled tests** | `tests/test_graph_loader.py` and `tests/test_phase1_skip_report.py` were rewritten to validate the new structure | The old tests hard-coded the 3-node seed and the compartment→report bug we were told to fix. |

---

## 3. How the graph is formatted

One JSON object per line. Every node carries the full field set required by the schema:

```json
{
  "id": "endometrium_assessment",
  "label": "Endometrium",
  "question": "Which process best characterizes the endometrium?",
  "description": "Assess endometrial gland-to-stroma ratio, gland crowding, cytologic atypia, inflammation, and surface architecture ...",
  "tier": "local_features",
  "node_kind": "local",
  "interaction": "single_select",
  "options": ["physiologic_cycling", "atrophic", "endometritis", "polyp", "hyperplasia", "carcinoma"],
  "edges": {"physiologic_cycling": "endometrium_cycle_phase", "atrophic": "synthesis_interpretation", "endometritis": "endometritis_type", "polyp": "synthesis_interpretation", "hyperplasia": "endometrial_hyperplasia_grade", "carcinoma": "endometrial_carcinoma_subtype"},
  "zoom_level": "20x",
  "visual_policy": "patch_retrieve",
  "requires_visual_evidence": true,
  "root": false,
  "is_leaf": false
}
```

### Field reference

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Stable unique node key (used by `edges` targets and traversal). |
| `label` | string | Human-readable name (the `uterus_graph.png` medical category). |
| `question` | string | The prompt shown to the VLM at this node. |
| `description` | string | **Retrieval text** — appended to `question` for CONCH/TITAN patch retrieval. Not displayed as the question. |
| `tier` | enum | `global_features` \| `local_features` \| `integration`. |
| `node_kind` | enum | `global` \| `compartment` \| `local` \| `integration` \| `report`. |
| `interaction` | enum | `single_select` \| `multi_select` \| `boolean` \| `free_text`. |
| `options` | list | Allowed answers. Every `single_select`/`boolean` option must have a matching `edges` key. |
| `edges` | map | `{answer: next_node_id}`. `multi_select` uses `__default__` to converge. |
| `zoom_level` | enum | `5x` \| `10x` \| `20x` — selects the offline CONCH patch pool. (No `40x`.) |
| `visual_policy` | enum | `thumbnail_only` \| `patch_retrieve` \| `both`. |
| `requires_visual_evidence` | bool | Whether the answer should be grounded in pixels. |
| `root` | bool | Exactly one node is the entry point (`organ_procedure`). |
| `is_leaf` | bool | Exactly one node terminates the walk (`report`); leaves have no `edges`. |

### Zoom + visual policy convention

| Node kind / role | `zoom_level` | `visual_policy` |
|------------------|--------------|-----------------|
| global / specimen | `5x` | `thumbnail_only` |
| compartment / gross | `10x` | `thumbnail_only` |
| local detail | `20x` | `patch_retrieve` |
| integration / report | `20x` | `both` |

---

## 4. Graph map (20 nodes)

```
organ_procedure (root, 5x)
  └─ compartment (10x)
       ├─ endometrium  → endometrium_assessment (20x)
       │     ├─ physiologic_cycling → endometrium_cycle_phase ─┐
       │     ├─ atrophic ──────────────────────────────────────┤
       │     ├─ endometritis → endometritis_type ───────────────┤
       │     ├─ polyp ───────────────────────────────────────────┤
       │     ├─ hyperplasia → endometrial_hyperplasia_grade ─────┤
       │     └─ carcinoma → endometrial_carcinoma_subtype         │
       │            → endometrial_carcinoma_grade                 │
       │            → background_endometrium → stage_extent ─┐    │
       ├─ myometrium → myometrium_assessment (20x)            │    │
       │     ├─ leiomyoma / leiomyosarcoma → smooth_muscle_tumor_assessment
       │     │        └─ malignant_two_or_more → stage_extent ┤    │
       │     │        └─ (benign/atypia/stump) ───────────────┼────┤
       │     ├─ endometrial_stromal_sarcoma → microscopic_pattern  │
       │     └─ adenomyosis / smooth_muscle_hyperplasia / degenerative_change ─┤
       ├─ junctional_zone → junctional_zone_assessment ──────────┤
       ├─ serosa_perimetrium → serosa_assessment ────────────────┤
       └─ mass_lesion → mass_histologic_type (20x)                │
              ├─ epithelial/mesenchymal/mixed → microscopic_pattern (20x)
              │        └─ cellular_features (multi) → stage_extent ┤
              ├─ gestational_trophoblastic → cellular_features ────┤
              └─ metastatic / other_lesion ───────────────────────┤
                                                                   │
   shared integration tail: stage_extent (20x both) ──────────────┘
        → synthesis_interpretation (20x both)
        → diagnosis (20x both)
        → report (leaf, free_text, 20x both)
```

All paths converge on `synthesis_interpretation → diagnosis → report`. There are no cycles
and no dead ends.

---

## 5. How it works at inference

### Deterministic traversal

The controller loads the graph once and walks it node-by-node. Navigation is keyed purely
on the VLM's answer:

- `JsonGraphStore.next(node_id, answer)` returns `edges[answer]`.
- `multi_select` nodes (e.g. `cellular_features`) converge via `edges["__default__"]`.
- The walk stops when it reaches the `is_leaf` node (`report`).

The LLM only ever classifies the current node; it cannot skip steps or leave the graph.
This reproducibility is what REG² evaluation (Binary Path Validity, Edge-F1) requires.

### Retrieval text

`question + description` form `node.retrieval_text` (see `graph/schema.py`). This string is
encoded by `TitanEncoder.encode_text()` and cosine-matched against the slide's
`patch_embeddings_{zoom_level}.pt` pool to pull the top-k patch images for the VLM. The
`description` therefore steers *which patches the VLM sees* — it is a retrieval signal, not a
displayed prompt. Retrieval only fires on `patch_retrieve` / `both` nodes;
`thumbnail_only` nodes (the `5x`/`10x` global + compartment nodes) skip it.

### Worked example — endometrioid adenocarcinoma

| Step | Node | Zoom / policy | Answer | Next node |
|------|------|---------------|--------|-----------|
| 1 | `organ_procedure` | 5x / thumbnail_only | `uterus_hysterectomy` | `compartment` |
| 2 | `compartment` | 10x / thumbnail_only | `endometrium` | `endometrium_assessment` |
| 3 | `endometrium_assessment` | 20x / patch_retrieve | `carcinoma` | `endometrial_carcinoma_subtype` |
| 4 | `endometrial_carcinoma_subtype` | 20x / patch_retrieve | `endometrioid` | `endometrial_carcinoma_grade` |
| 5 | `endometrial_carcinoma_grade` | 20x / patch_retrieve | `grade_1` | `background_endometrium` |
| 6 | `background_endometrium` | 20x / patch_retrieve | `hyperplastic_background` | `stage_extent` |
| 7 | `stage_extent` | 20x / both | `superficial_invasion` | `synthesis_interpretation` |
| 8 | `synthesis_interpretation` | 20x / both | `definitive` | `diagnosis` |
| 9 | `diagnosis` | 20x / both | `malignant` | `report` |
| 10 | `report` | 20x / both (leaf) | CAP report text | — stop |

Switching the case (e.g. a leiomyosarcoma) changes only the answers and therefore the path;
the same machinery applies.

---

## 6. How it follows the project structure

- **Owner lane:** the graph and `graph/loader.py` are the DOGA lane (PROJECT_OVERVIEW §7).
  This work stays inside that lane — no changes to vision, retrieval, or memory code.
- **Schema is read-only ground truth:** the node fields, tiers, and enums come from
  `graph/schema.py` and PROJECT_OVERVIEW §5. We populated the JSONL to that contract rather
  than changing the schema.
- **Loader/validator unchanged:** `graph/loader.py` parses and validates the file. The graph
  satisfies all of its invariants (edge targets exist, leaves have no edges, every
  select-option has an edge).
- **Offline cache contract:** `zoom_level` values are restricted to the pools that actually
  exist on disk (`patch_embeddings_{5x,10x,20x}.pt`), per PROJECT_OVERVIEW §Phase 0.
- **Tests:** validation lives in `tests/test_graph_loader.py`; traversal coupling is checked
  by `tests/test_controller.py` and `tests/test_phase1_skip_report.py`.

---

## 7. How it contributes to the whole project

The graph is the **navigation backbone of Phase 1** and the scaffold for Phase 2 and eval:

- **Phase 1 (graph traversal):** each visited node contributes a `(question, answer)` step to
  `runs/{slide_id}/cot_chain.json`. `zoom_level` routes each node to its CONCH pool and
  `description` drives patch retrieval — the graph is the project's "graph-as-MST"
  magnification policy (PROJECT_OVERVIEW §10, MMNavAgent borrow).
- **Phase 2 (report generation):** the accumulated chain reaches the `report` leaf, where the
  report LLM (MedGemma) synthesizes the CAP-format pathology report.
- **Evaluation (REG²):** because navigation is deterministic and graph-owned, the produced
  reasoning chains are reproducible and directly comparable against ground-truth chains
  (Binary Path Validity, Edge-F1, MESS).

In short: the graph turns "a WSI" into a structured, auditable chain of visually-grounded
diagnostic decisions that the rest of the pipeline consumes.

---

## 8. Validation

| Check | Guarantee |
|-------|-----------|
| Exactly one `root` | `organ_procedure` |
| Exactly one `is_leaf` | `report` (no outgoing edges) |
| Every `single_select`/`boolean` option has an edge | enforced by `validate_graph` |
| Every edge target exists | enforced by `validate_graph` |
| No `40x` zoom levels | all nodes are `5x`/`10x`/`20x` |
| Zoom/visual policy matches node kind | per the §3 convention table |
| Traversal reaches `report` | confirmed with `DummyBackend` walk |
| `compartment` does not jump to `report` | confirmed (routes into local work-ups) |

Run before a PR:

```bash
uv run pytest tests/test_graph_loader.py -q
```

---

## 9. Future work

- **40x flip:** if a 40x pool is added offline, only change `zoom_level` on the nuclei/mitosis
  nodes (e.g. `cellular_features`, `endometrial_carcinoma_grade`). Stable ids make this a
  one-field edit.
- **Deeper subtyping:** mass-lesion and mesenchymal branches can be expanded with more
  WHO-aligned subtypes without touching the integration tail.
- **Ontology mirror:** an optional `data/graph/ontology_graph.jsonl` full drawio export can be
  added later (PROJECT_OVERVIEW §4).
