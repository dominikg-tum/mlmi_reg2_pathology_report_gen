"""P1 ablation: TITAN slide-level embedding + evidence patch PNGs for the VLM."""

from __future__ import annotations

from pathlib import Path

from graph.schema import Node
from vision.backends import VisualBundle
from vision.cache import SlideCache


class SlideEmbedProvider:
    """Baseline 2: offline TITAN slide embedding + tissue patch images for Qwen.

    Raw 768-d vectors cannot be fed to a VLM directly. We attach:
    - whole-slide thumbnail (global context)
    - 3 evidence patch PNGs saved during offline encoding
    - slide_embedding in VisualBundle for logging / future projector
    """

    def __init__(self, cache_root: Path | None = None):
        self.cache_root = cache_root

    def for_node(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle:
        bundle = VisualBundle(metadata={"visual": "slide_embed"})
        if slide_cache is None:
            return bundle

        if slide_cache.thumbnail_path and slide_cache.thumbnail_path.exists():
            bundle.thumbnail_path = slide_cache.thumbnail_path

        emb = slide_cache.load_slide_embedding()
        if emb is not None:
            bundle.slide_embedding = emb
            bundle.metadata["slide_embedding_dim"] = int(emb.shape[0])

        evidence = slide_cache.evidence_patch_paths()
        bundle.patch_paths = evidence
        bundle.metadata["evidence_patch_count"] = len(evidence)
        return bundle
