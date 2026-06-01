# Memory (NICK)

| `--memory` | Status |
|------------|--------|
| `flat` | Episodic only (default) — `memory/episodic.py` |
| `hipporag2` | Stub — implement `hipporag2.py` |
| `graphrag` | Stub — implement `graphrag.py` |

Build index from **train split only** (no test leakage). Semantic memory never routes the graph.

Wire via `agent/memory.py` → `CaseMemory.retrieve_context()`.
