"""Offline: .svs -> thumbnail PNG per slide (P1, no TITAN). DOMI implements."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError(
        "DOMI: openslide read -> downsample to max_edge_px from configs/vision.yaml -> "
        "save {cache_root}/{slide_id}/thumbnail.png"
    )


if __name__ == "__main__":
    main()
