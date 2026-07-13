# Memory (NICK)

| `--memory`  | Status                                         |
|-------------|------------------------------------------------|
| `flat`      | Episodic only (default) — `memory/episodic.py` |
| `hipporag2` | Embedding fallback over train CoT steps — `build_hipporag_index.py` |
| `hybridrag` | Chroma + BM25 over train reports **+ reference chunks** — `build_hybridrag_index.py` |
| `graphrag`  | Stub — implement `graphrag.py`                 |

Build index from **train split only** (no test leakage). Semantic memory never routes the graph.

Reference corpus: `data/memory/reference/**/*.jsonl` — see [`data/memory/reference/README.md`](../data/memory/reference/README.md).

```bash
python -m scripts.memory.build_hipporag_index --split train
python -m scripts.memory.build_hybridrag_index --split train
# After editing reference chunks:
python -m scripts.memory.build_hybridrag_index --split train --force-rebuild
```

`--memory hybridrag` auto-loads from `data/memory/hybridrag_manifest.json` when the Chroma store exists.
