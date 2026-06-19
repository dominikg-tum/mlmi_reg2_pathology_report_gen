"""P1 default: whole-slide thumbnail (no TITAN)."""

from __future__ import annotations

from pathlib import Path

from graph.schema import Node
from vision.backends import VisualBundle
from vision.cache import SlideCache
def _resolve_wsi_path(
    slide_cache: SlideCache | None,
    *,
    wsi_path: Path | None = None,
    wsi_data_dir: Path | None = None,
) -> Path | None:
    if wsi_path is not None and wsi_path.exists():
        return wsi_path
    if slide_cache is None or wsi_data_dir is None:
        return None
    matches = list(wsi_data_dir.rglob(slide_cache.slide_id))
    return matches[0] if matches else None


def _bundle_from_retrieved(
    retrieved: list,
    slide_cache: SlideCache,
    *,
    out_subdir: str = "retrieved",
    include_thumbnail: bool = False,
    metadata: dict | None = None,
) -> VisualBundle:
    bundle = VisualBundle(metadata=dict(metadata or {"visual": "patch_retrieve"}))
    if include_thumbnail and slide_cache.thumbnail_path:
        bundle.thumbnail_path = slide_cache.thumbnail_path

    out_dir = (slide_cache.cache_dir or Path(".")) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    meta_rows: list[dict] = []

    for i, rp in enumerate(retrieved):
        row = {
            "index": rp.index,
            "level": rp.level,
            "coord": rp.coord,
            "parent_coord": rp.parent_coord,
            "parent_index": rp.parent_index,
            "parent_level": rp.parent_level,
            "grandparent_coord": rp.grandparent_coord,
            "grandparent_index": rp.grandparent_index,
            "grandparent_level": rp.grandparent_level,
            "similarity": rp.similarity,
        }
        if rp.patch_image is not None:
            p = out_dir / f"patch_{i}_{rp.level}.jpg"
            rp.patch_image.convert("RGB").save(p, format="JPEG", quality=90)
            paths.append(p)
            row["patch_path"] = str(p)
        if rp.parent_image is not None:
            pl = rp.parent_level or "parent"
            pp = out_dir / f"patch_{i}_parent_{pl}.jpg"
            rp.parent_image.convert("RGB").save(pp, format="JPEG", quality=90)
            row["parent_patch_path"] = str(pp)
        if rp.grandparent_image is not None:
            gl = rp.grandparent_level or "grandparent"
            gp = out_dir / f"patch_{i}_grandparent_{gl}.jpg"
            rp.grandparent_image.convert("RGB").save(gp, format="JPEG", quality=90)
            row["grandparent_patch_path"] = str(gp)
        meta_rows.append(row)

    bundle.patch_paths = paths
    bundle.metadata["retrieved_patches"] = meta_rows
    return bundle


class ThumbnailProvider:
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
        """Ablation baseline: whole-slide thumbnail on every node (ignores graph policy)."""
        _ = node, query, retriever
        thumb = slide_cache.thumbnail_path if slide_cache else None
        return VisualBundle(thumbnail_path=thumb, metadata={"visual": "thumbnail"})


class NoneVisualProvider:
    """Text-only debug / oracle runs."""

    def for_node(self, node, slide_cache=None, *, query: str, retriever=None) -> VisualBundle:
        return VisualBundle(metadata={"visual": "none"})


class PatchRetrieveProvider:
    """Backward-compatible alias for graph-policy visual routing."""

    def __init__(
        self,
        cache_root: Path | None = None,
        *,
        wsi_path: Path | None = None,
        wsi_data_dir: Path | None = None,
    ):
        from vision.graph_visual import GraphPolicyVisualProvider

        self._delegate = GraphPolicyVisualProvider(
            cache_root,
            wsi_path=wsi_path,
            wsi_data_dir=wsi_data_dir,
        )

    def for_node(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle:
        return self._delegate.for_node(
            node, slide_cache, query=query, retriever=retriever
        )
