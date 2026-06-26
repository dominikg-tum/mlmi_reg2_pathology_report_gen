"""UNI2-h patch encoder adapter.

The upstream UNI repository is expected to be cloned locally, e.g. /Volumes/Xun/UNI.
This module keeps the dependency import lazy so ordinary tests and frontend runs do
not import timm/torchvision or download weights.
"""

from __future__ import annotations

import sys
import os
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class UNI2Encoder:
    """Encode PIL image patches with MahmoodLab UNI2-h."""

    def __init__(
        self,
        *,
        repo_path: Path | str,
        model_name: str = "uni2-h",
        weights_path: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser()
        self.model_name = model_name
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self.device = device
        self._model = None
        self._transform = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.repo_path.exists():
            raise FileNotFoundError(f"UNI repo not found: {self.repo_path}")
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        import torch
        import huggingface_hub
        from uni.get_encoder.get_encoder import get_encoder

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        assets_dir = _resolve_assets_dir(self.weights_path, self.model_name)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        original_login = huggingface_hub.login

        def _login_non_interactive(*_args, **_kwargs):
            if not token:
                raise RuntimeError(
                    "UNI2 weights require Hugging Face access. Set HF_TOKEN or pass "
                    "--uni2-weights-path pointing to a local checkpoint directory/file."
                )
            return original_login(token=token, add_to_git_credential=False)

        huggingface_hub.login = _login_non_interactive
        try:
            encoder_kwargs = {
                "enc_name": self.model_name,
                "img_resize": 224,
                "center_crop": True,
                "device": dev,
            }
            if assets_dir is not None:
                encoder_kwargs["assets_dir"] = str(assets_dir)
            model, transform = get_encoder(**encoder_kwargs)
        finally:
            huggingface_hub.login = original_login
        if model is None or transform is None:
            raise RuntimeError(f"Could not load UNI encoder {self.model_name!r}")
        self._model = model
        self._transform = transform

    def encode_patches(self, patch_images: list[Any], *, batch_size: int = 16) -> np.ndarray:
        self._ensure_loaded()
        import torch

        feats: list[np.ndarray] = []
        for start in range(0, len(patch_images), batch_size):
            batch = patch_images[start : start + batch_size]
            tensors = torch.stack([self._transform(img.convert("RGB")) for img in batch])
            tensors = tensors.to(self.device)
            with torch.inference_mode():
                out = self._model(tensors)
            feats.append(out.detach().float().cpu().numpy())
        if not feats:
            return np.zeros((0, 1536), dtype=np.float32)
        return np.vstack(feats).astype(np.float32)


def _resolve_assets_dir(weights_path: Path | None, model_name: str) -> Path | None:
    """Return the UNI assets_dir containing <model_name>/pytorch_model.bin."""
    if weights_path is None:
        return None
    if weights_path.is_file():
        if weights_path.name != "pytorch_model.bin":
            raise ValueError(f"UNI2 checkpoint file must be pytorch_model.bin: {weights_path}")
        if weights_path.parent.name == model_name:
            return weights_path.parent.parent
        if weights_path.parent.name.lower() == model_name.lower():
            return _symlink_assets_dir(weights_path.parent, model_name)
        raise ValueError(
            f"UNI2 checkpoint file must live under a {model_name}/ directory: {weights_path}"
        )
    if weights_path.name == model_name:
        return weights_path.parent
    if weights_path.name.lower() == model_name.lower():
        return _symlink_assets_dir(weights_path, model_name)
    if (weights_path / "pytorch_model.bin").exists():
        return _symlink_assets_dir(weights_path, model_name)
    if (weights_path / model_name / "pytorch_model.bin").exists():
        return weights_path
    return weights_path.parent


def _symlink_assets_dir(weights_dir: Path, model_name: str) -> Path:
    digest = hashlib.sha1(str(weights_dir).encode("utf-8")).hexdigest()[:12]
    assets_dir = Path("/tmp") / f"mlmi_uni2_assets_{digest}"
    link = assets_dir / model_name
    assets_dir.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() == weights_dir.resolve():
            return assets_dir
        link.unlink()
    link.symlink_to(weights_dir, target_is_directory=True)
    return assets_dir


def mean_pool_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Simple WSI embedding baseline: mean-pool patch embeddings."""
    if embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32).mean(axis=0).astype(np.float32)
