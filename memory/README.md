# Memory (NICK)

| `--memory` / baseline | Status |
|-----------------------|--------|
| `flat` | Episodic only (default) — `memory/episodic.py` |
| `hipporag2` / `b1` | Embedding fallback over train CoT steps — `build_hipporag_index.py` |
| `hybridrag` / `b2` | Chroma + BM25 over **train reports only** (nocap) |
| `hybridrag_cap` / `b2_cap` | Same HybridRAG + CAP/WHO reference chunks |
| `graphrag` | Stub — implement `graphrag.py` |

Build index from **train split only** (no test leakage). Semantic memory never routes the graph.

Dual ablation stores (build once, switch via baseline flag — no rebuild when switching):

| Variant | Chroma (paths.yaml) | Manifest | Contents |
|---------|---------------------|----------|----------|
| `nocap` | `rag.chroma_db_storage_nocap` | `hybridrag_manifest_nocap.json` | train reports |
| `cap` | `rag.chroma_db_storage_cap` | `hybridrag_manifest_cap.json` | reports + `data/memory/reference/**/*.jsonl` |

Reference corpus: `data/memory/reference/**/*.jsonl` — see [`data/memory/reference/README.md`](../data/memory/reference/README.md).

```bash
# Build both stores (cluster)
VARIANT=both FORCE_REBUILD=1 sbatch --export=NONE,VARIANT,FORCE_REBUILD \
  --job-name=path-hybridrag-both scripts/cluster/build_hybridrag_index.sh

# Or locally / in container
python -m scripts.memory.build_hybridrag_index --variant nocap --force-rebuild
python -m scripts.memory.build_hybridrag_index --variant cap --force-rebuild

# Run ablation
BASELINE=b2 sbatch ... scripts/cluster/run_baseline_batch.sh      # reports only
BASELINE=b2_cap sbatch ... scripts/cluster/run_baseline_batch.sh  # + CAP refs
```

`--memory hybridrag` auto-loads the **nocap** manifest. `--memory hybridrag_cap` loads the CAP store.
Rebuild only after changing reports or CAP chunks (`--force-rebuild`).
