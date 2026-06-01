"""Offline: extract patches + TITAN encode -> embeddings_{level}.pt. DOMI implements."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError(
        "DOMI: sbatch job — patch at mag band, encode with vision/encoders/titan.py, "
        "write embeddings_{level}.pt + coords_{level}.pt under cache_root"
    )


if __name__ == "__main__":
    main()
