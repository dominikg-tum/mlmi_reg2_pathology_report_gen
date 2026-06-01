"""B1 baseline: WSI thumbnail → report without reasoning chain."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct report generation (no graph)")
    parser.add_argument("--slide-id", required=True)
    args = parser.parse_args()
    raise NotImplementedError(
        "DOMI/XUN: single VLM call with thumbnail only; compare to run_agent chain output"
    )


if __name__ == "__main__":
    main()
