"""TITAN + CONCH v1.5 offline encoder (cluster GPU jobs only)."""

from __future__ import annotations

from typing import Any

import numpy as np


class TitanEncoder:
    """Frozen CONCH patch encoder + TITAN slide aggregator from MahmoodLab/TITAN."""

    def __init__(self, model_id: str = "MahmoodLab/TITAN", device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._titan = None
        self._conch = None
        self._transform = None

    def _ensure_loaded(self) -> None:
        if self._titan is not None:
            return
        import torch
        from transformers import AutoModel

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self._titan = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(dev)
        self._titan.eval()
        self._conch, self._transform = self._titan.return_conch()
        self._conch = self._conch.to(dev)
        self._conch.eval()

    def encode_patches(self, patch_images: list, *, batch_size: int = 32) -> np.ndarray:
        """Return [N, D] CONCH/TITAN patch features."""
        self._ensure_loaded()
        import torch

        feats: list[np.ndarray] = []
        for start in range(0, len(patch_images), batch_size):
            batch = patch_images[start : start + batch_size]
            tensors = torch.stack([self._transform(img) for img in batch])
            tensors = tensors.to(self.device, dtype=torch.float32)
            with torch.inference_mode():
                out = self._conch(tensors)
            if isinstance(out, dict):
                out = out.get("embeddings", out.get("pooler_output", next(iter(out.values()))))
            feats.append(out.float().cpu().numpy())
        if not feats:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(feats).astype(np.float32)

    def encode_slide(
        self,
        patch_features: np.ndarray,
        coords: np.ndarray,
        patch_size_lv0: int,
    ) -> np.ndarray:
        """Aggregate patch features into one slide-level embedding [D]."""
        self._ensure_loaded()
        import torch

        if patch_features.size == 0:
            raise ValueError("Cannot encode slide from zero patches")
        features = torch.from_numpy(patch_features).float()
        coord_t = torch.from_numpy(np.asarray(coords, dtype=np.int64))
        with torch.inference_mode():
            slide = self._titan.encode_slide_from_patch_features(
                features.to(self.device),
                coord_t.to(self.device),
                int(patch_size_lv0),
            )
        vec = slide.float().cpu().numpy().reshape(-1)
        return vec.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Text encoder in the shared TITAN space (for retrieval / zero-shot)."""
        self._ensure_loaded()
        import torch

        # TITAN.encode_text expects tokenized input_ids (same as zero_shot_classifier),
        # not raw strings. Passing a list crashes on input_ids[:, :-1].
        tokenizer = self._titan.text_encoder.tokenizer
        tokens = tokenizer([text]).to(self.device)
        with torch.inference_mode():
            out = self._titan.encode_text(tokens, normalize=True)
        if isinstance(out, torch.Tensor):
            vec = out[0]
        else:
            vec = out[0] if isinstance(out, (list, tuple)) else out
        return vec.float().cpu().numpy().reshape(-1).astype(np.float32)


class TitanPatchEncoder:
    """Thin wrapper matching the PatchEncoder protocol."""

    def __init__(self, model_id: str = "MahmoodLab/TITAN", device: str | None = None):
        self._inner = TitanEncoder(model_id=model_id, device=device)

    def encode_patches(self, patch_images: list) -> np.ndarray:
        return self._inner.encode_patches(patch_images)
