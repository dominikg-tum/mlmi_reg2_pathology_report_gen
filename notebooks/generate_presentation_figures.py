#!/usr/bin/env python3
"""Generate presentation-ready baseline evaluation figures.

Run from repo root:
    python notebooks/generate_presentation_figures.py

Outputs flat copies to notebooks/figures/ (for slides) plus a timestamped subfolder.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from notebooks.figure_utils import generate_all_figures


def main() -> None:
    fig_dir = generate_all_figures(repo=REPO)
    flat_dir = REPO / "notebooks" / "figures"
    print(f"Writing figures → {fig_dir} (+ flat copies in {flat_dir})")
    print("\nGenerated figures:")
    for p in sorted(flat_dir.glob("*.png")):
        print(f"  {p.name}")
    print(f"\nNarrative: {REPO / 'notebooks' / 'PRESENTATION_NARRATIVE.md'}")


if __name__ == "__main__":
    main()
