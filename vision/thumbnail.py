"""P1 default: whole-slide thumbnail (no TITAN)."""

from __future__ import annotations

from pathlib import Path

from graph.schema import Node, VisualPolicy
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
) -> VisualBundle:
    bundle = VisualBundle(metadata={"visual": "patch_retrieve"})
    if slide_cache.thumbnail_path:
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
            p = out_dir / f"patch_{i}_{rp.level}.png"
            rp.patch_image.save(p)
            paths.append(p)
            row["patch_path"] = str(p)
        if rp.parent_image is not None:
            pl = rp.parent_level or "parent"
            pp = out_dir / f"patch_{i}_parent_{pl}.png"
            rp.parent_image.save(pp)
            paths.append(pp)
            row["parent_patch_path"] = str(pp)
        if rp.grandparent_image is not None:
            gl = rp.grandparent_level or "grandparent"
            gp = out_dir / f"patch_{i}_grandparent_{gl}.png"
            rp.grandparent_image.save(gp)
            paths.append(gp)
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
        thumb = slide_cache.thumbnail_path if slide_cache else None
        bundle = VisualBundle(thumbnail_path=thumb, metadata={"visual": "thumbnail"})
        if node.visual_policy == VisualPolicy.BOTH and retriever is not None and slide_cache:
            try:
                wsi = _resolve_wsi_path(
                    slide_cache, wsi_path=self.wsi_path, wsi_data_dir=self.wsi_data_dir
                )
                retrieved = retriever.retrieve(
                    node.retrieval_text,
                    slide_cache,
                    level=node.mag_band,
                    wsi_path=wsi,
                    return_images=wsi is not None,
                    tier=node.tier.value,
                    node_kind=node.node_kind.value,
                )
                patch_bundle = _bundle_from_retrieved(retrieved, slide_cache, out_subdir="retrieved_both")
                bundle.patch_paths = patch_bundle.patch_paths
                bundle.metadata.update(patch_bundle.metadata)
            except (RuntimeError, NotImplementedError, FileNotFoundError):
                pass
        return bundle


class NoneVisualProvider:
    """Text-only debug / oracle runs."""

    def for_node(self, node, slide_cache=None, *, query: str, retriever=None) -> VisualBundle:
        return VisualBundle(metadata={"visual": "none"})


class PatchRetrieveProvider:
    """P2: top-K patches from offline cache via retriever."""

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
        if retriever is None or slide_cache is None or not node.needs_patch_retrieval():
            return VisualBundle(metadata={"visual": "patch_retrieve"})

        wsi = _resolve_wsi_path(
            slide_cache, wsi_path=self.wsi_path, wsi_data_dir=self.wsi_data_dir
        )
        retrieved = retriever.retrieve(
            node.retrieval_text,
            slide_cache,
            level=node.mag_band,
            wsi_path=wsi,
            return_images=wsi is not None,
            tier=node.tier.value,
            node_kind=node.node_kind.value,
        )
        return _bundle_from_retrieved(retrieved, slide_cache)
