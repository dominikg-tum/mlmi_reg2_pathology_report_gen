# HybridRAG reference corpus

Curated pathology knowledge chunks indexed **alongside** train-split case reports.
Use for diagnostic criteria and CAP-aligned reporting language; case reports remain
the primary source for real-world phrasing examples.

## Layout

```text
data/memory/reference/
  uterus/chunks.jsonl    # seed chunks (CAP/WHO-aligned uterus graph)
  <topic>/chunks.jsonl   # add more topics as needed
```

All `*.jsonl` files under this tree are loaded recursively at index build time.

## Chunk schema (one JSON object per line)

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Stable chunk id (used as `slide_id` in metadata for display) |
| `title` | yes | Short section title |
| `text` | yes | Chunk body indexed by Chroma + BM25 |
| `source` | yes | Provenance string (e.g. `CAP Endometrium v5.1`) |
| `source_type` | yes | Must be `"reference"` |
| `topic` | no | Organ/topic tag (`endometrium`, `myometrium`, `uterus`) |
| `graph_nodes` | no | List of `execution_graph.jsonl` node ids this chunk supports |
| `tier` | no | `global_features` \| `local_features` \| `integration` (rerank hint) |

## Adding CAP PDF content

1. Download [CAP Endometrium v5.1](https://documents.cap.org/protocols/Uterus_5.1.0.0.REL.CAPCP.pdf) and [CAP Uterine Sarcoma v4.4](https://documents.cap.org/protocols/Uterus.Sarc_4.4.0.0.REL_CAPCP.pdf).
2. Split by protocol section (Histologic Type, Grade, Myometrial Invasion, …).
3. Append one JSON object per section to `uterus/chunks.jsonl`.
4. Rebuild the **cap** HybridRAG store only (`--variant cap --force-rebuild`). Keep the nocap store unchanged for the `b2` vs `b2_cap` ablation.

```bash
python -m scripts.memory.build_hybridrag_index --variant cap --force-rebuild
# or both:
VARIANT=both FORCE_REBUILD=1 sbatch --export=NONE,VARIANT,FORCE_REBUILD \
  --job-name=path-hybridrag-both scripts/cluster/build_hybridrag_index.sh
```

Baselines: `--baseline b2` (reports only) vs `--baseline b2_cap` (reports + this reference corpus).

## Retrieval behaviour

At `local_features` and `integration` graph tiers, reference chunks are ranked ahead of
case reports when scores are similar. Global nodes keep the ensemble order (case reports
often carry specimen-level context).
