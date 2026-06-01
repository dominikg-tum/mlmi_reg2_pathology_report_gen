# Retrieval (DOMI)

- **P1:** no retriever (`--retriever none`)
- **P2:** offline `embeddings_*.pt` then `--retriever titan_cosine`

`graph_guided` wraps any inner retriever; controller passes `node.retrieval_level`.

Ablation: add new class implementing `PatchRetriever` in this package.
