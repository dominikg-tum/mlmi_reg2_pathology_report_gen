"""Per-node visual routing driven by graph ``visual_policy``."""

from __future__ import annotations

from pathlib import Path

from graph.schema import Node, VisualPolicy
from vision.backends import VisualBundle
from vision.cache import SlideCache
from vision.thumbnail import _bundle_from_retrieved, _resolve_wsi_path


def _policy_metadata(policy: VisualPolicy) -> dict:
    return {"visual": policy.value, "visual_policy": policy.value}


class GraphPolicyVisualProvider:
    """Route visual evidence per ``node.visual_policy``.

    | Policy            | VLM input                                      |
    |-------------------|------------------------------------------------|
    | ``thumbnail_only``| Whole-slide thumbnail                          |
    | ``patch_retrieve``| Top-k retrieved patches at ``node.zoom_level`` |
    | ``both``          | Thumbnail + retrieved patches                  |
    """

    def __init__(
        self,
        cache_root: Path | None = None,
        *,
        wsi_path: Path | None = None,
        wsi_data_dir: Path | None = None,
    ):
        self.cache_root = cache_root
        self.wsi_path = wsi_path
        self.wsi_data_dir = wsi_data_dir

    def for_node(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle:
        policy = node.visual_policy
        if policy == VisualPolicy.THUMBNAIL_ONLY:
            return self._thumbnail_only(slide_cache)
        if policy == VisualPolicy.PATCH_RETRIEVE:
            return self._patch_retrieve(node, slide_cache, retriever=retriever)
        if policy == VisualPolicy.BOTH:
            return self._both(node, slide_cache, retriever=retriever)
        raise ValueError(f"Unknown visual_policy: {policy!r}")

    def _thumbnail_only(self, slide_cache: SlideCache | None) -> VisualBundle:
        thumb = slide_cache.thumbnail_path if slide_cache else None
        return VisualBundle(
            thumbnail_path=thumb,
            metadata=_policy_metadata(VisualPolicy.THUMBNAIL_ONLY),
        )

    def _patch_retrieve(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        retriever,
    ) -> VisualBundle:
        if retriever is None or slide_cache is None:
            return VisualBundle(metadata=_policy_metadata(VisualPolicy.PATCH_RETRIEVE))
        retrieved = self._retrieve(node, slide_cache, retriever)
        return _bundle_from_retrieved(
            retrieved,
            slide_cache,
            out_subdir="retrieved",
            include_thumbnail=False,
            metadata=_policy_metadata(VisualPolicy.PATCH_RETRIEVE),
        )

    def _both(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        retriever,
    ) -> VisualBundle:
        thumb = slide_cache.thumbnail_path if slide_cache else None
        bundle = VisualBundle(
            thumbnail_path=thumb,
            metadata=_policy_metadata(VisualPolicy.BOTH),
        )
        if retriever is None or slide_cache is None:
            return bundle
        try:
            retrieved = self._retrieve(node, slide_cache, retriever)
            patch_bundle = _bundle_from_retrieved(
                retrieved,
                slide_cache,
                out_subdir="retrieved_both",
                include_thumbnail=False,
                metadata=_policy_metadata(VisualPolicy.BOTH),
            )
            bundle.patch_paths = patch_bundle.patch_paths
            bundle.metadata["retrieved_patches"] = patch_bundle.metadata.get(
                "retrieved_patches", []
            )
        except (RuntimeError, NotImplementedError, FileNotFoundError):
            pass
        return bundle

    def _retrieve(self, node: Node, slide_cache: SlideCache, retriever):
        wsi = _resolve_wsi_path(
            slide_cache,
            wsi_path=self.wsi_path,
            wsi_data_dir=self.wsi_data_dir,
        )
        return retriever.retrieve(
            node.retrieval_text,
            slide_cache,
            level=node.mag_band,
            wsi_path=wsi,
            return_images=wsi is not None,
            tier=node.tier.value,
            node_kind=node.node_kind.value,
        )
