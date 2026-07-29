"""Timestamped baseline run paths — avoid overwriting predictions and figures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Human-readable baseline keys -> predictions file stem (without split / run_id).
BASELINE_PRED_STEMS: dict[str, str] = {
    "a": "baseline_a",
    "b1": "baseline_b1",
    "b2": "baseline_b2",
    "patch_retrieve": "baseline_patch_retrieve",
    "patch_retrieve_fullpool": "baseline_patch_retrieve_fullpool",
    "patch_retrieve_smoke": "patch_retrieve_smoke",
}


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def predictions_stem(*, baseline: str | None, runs_subdir: str | None, split: str) -> str:
    if baseline and baseline.lower() in BASELINE_PRED_STEMS:
        return BASELINE_PRED_STEMS[baseline.lower()]
    if runs_subdir:
        return runs_subdir.replace("/", "_")
    return f"baseline_{split}"


def timestamped_predictions_path(runs_dir: Path, *, split: str, stem: str, run_id: str) -> Path:
    return runs_dir / f"predictions_{split}_{stem}_{run_id}.jsonl"


def timestamped_runs_subdir(base_subdir: str, run_id: str) -> str:
    return f"{base_subdir}_{run_id}"


def legacy_predictions_path(runs_dir: Path, *, split: str, stem: str) -> Path:
    return runs_dir / f"predictions_{split}_{stem}.jsonl"


def latest_predictions_link(runs_dir: Path, *, split: str, stem: str) -> Path:
    return runs_dir / f"predictions_{split}_{stem}_latest.jsonl"


def resolve_predictions_path(
    runs_dir: Path,
    *,
    split: str,
    stem: str,
    run_id: str | None = None,
) -> Path | None:
    """Return a predictions file: explicit run_id, latest symlink, or legacy fixed name."""
    if run_id:
        path = timestamped_predictions_path(runs_dir, split=split, stem=stem, run_id=run_id)
        return path if path.exists() else None
    link = latest_predictions_link(runs_dir, split=split, stem=stem)
    if link.is_symlink():
        target = runs_dir / Path(os_readlink(link))
        return target if target.exists() else None
    if link.exists() and not link.is_symlink():
        return link
    legacy = legacy_predictions_path(runs_dir, split=split, stem=stem)
    return legacy if legacy.exists() else None


def os_readlink(path: Path) -> str:
    return path.readlink()


def update_latest_symlink(link_path: Path, target: Path) -> None:
    """Point ``*_latest.jsonl`` at the newest timestamped predictions file."""
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target.name)


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def discover_baseline_predictions(
    runs_dir: Path,
    *,
    split: str = "test",
    run_id: str | None = None,
) -> dict[str, Path]:
    """Map display labels to resolved prediction paths (latest or specific run_id)."""
    labels = {
        "a": "A (flat thumbnail)",
        "b1": "B1 (HippoRAG2)",
        "b2": "B2 (HybridRAG)",
        "patch_retrieve": "Patch (k=100 centroids)",
        "patch_retrieve_fullpool": "Patch (full pool)",
    }
    out: dict[str, Path] = {}
    for key, label in labels.items():
        stem = BASELINE_PRED_STEMS[key]
        path = resolve_predictions_path(runs_dir, split=split, stem=stem, run_id=run_id)
        if path is not None:
            out[label] = path
    return out


def timestamped_fig_dir(repo_root: Path, run_id: str | None = None) -> Path:
    rid = run_id or run_timestamp()
    return repo_root / "notebooks" / "figures" / rid
