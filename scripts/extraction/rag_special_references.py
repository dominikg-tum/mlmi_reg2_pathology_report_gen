"""Chunk CAP Endometrium v5.1 explanatory notes into HybridRAG reference JSONL."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import pdfplumber

# scripts/extraction/rag_special_references.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = REPO_ROOT / "data" / "raw" / "cap_protocols" / "Uterus_5.1.0.0.REL.CAPCP.pdf"
OUTPUT_DIR = REPO_ROOT / "data" / "memory" / "reference" / "cap_endometrium"

# CAP explanatory note letter -> uterus graph node(s) it should boost at retrieval
NOTE_TO_GRAPH_NODES = {
    "A": [],  # Clinical History
    "B": ["organ_procedure"],
    "C": ["endometrial_carcinoma_subtype"],
    "D": ["endometrial_carcinoma_grade"],
    "E": [],  # Molecular Type
    "F": ["stage_extent"],
    "G": ["stage_extent", "serosa_assessment"],
    "H": ["stage_extent"],
    "I": [],  # Peritoneal Washings
    "J": ["stage_extent", "cellular_features"],
    "K": [],  # Margins
    "L": [],  # Lymph Node Status
    "M": ["stage_extent"],
    "N": ["stage_extent"],
    "O": ["endometrial_hyperplasia_grade", "diagnosis"],
}


def extract_pdf_text(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def strip_page_footers(text: str) -> str:
    # recurring "<page num>\nCAP Uterus_5.1.0.0.REL_CAPCP\nApproved" block on every page
    pattern = re.compile(
        r"^\d+\r?\nCAP Uterus_5\.1\.0\.0\.REL_CAPCP\r?\nApproved\r?\n",
        re.MULTILINE,
    )
    return pattern.sub("", text)


def extract_explanatory_notes(text: str) -> dict[str, dict[str, str]]:
    marker = "Explanatory Notes"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("Marker 'Explanatory Notes' not found in text")
    notes_text = text[idx + len(marker):]

    header_pattern = re.compile(r"^([A-Z])\.\s+(.+)$", re.MULTILINE)
    headers = list(header_pattern.finditer(notes_text))

    sections: dict[str, dict[str, str]] = {}
    for i, match in enumerate(headers):
        letter = match.group(1)
        title = match.group(2).strip()
        body_start = match.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(notes_text)
        body = notes_text[body_start:body_end]

        # reference lists are pure citation noise for the embeddings, drop them
        body = body.split("\nReferences\n")[0].strip()
        # inline citation superscripts like "laparotomy.1" or "clefts.2,3"
        body = re.sub(r"(?<=[a-zA-Z\)])\.(\d{1,2}(,\d{1,2})*)\b", ".", body)

        if not body.strip():
            raise ValueError(f"Note {letter} ('{title}') has an empty body after cleanup")

        sections[letter] = {"title": title, "body": body}
    return sections


def sections_to_chunks(sections: dict, mapping: dict) -> list[dict]:
    chunks = []
    for letter, data in sections.items():
        raw_nodes = mapping.get(letter, [])
        chunks.append({
            "id": f"cap_endo_v51_{letter.lower()}",
            "title": data["title"],
            "text": data["body"],
            "source": "CAP Endometrium v5.1 (Uterus_5.1.0.0.REL_CAPCP)",
            "source_type": "reference",
            "topic": data["title"],
            "graph_nodes": raw_nodes or ["unmapped"],
            "_is_mapped": bool(raw_nodes),
        })
    return chunks


def write_jsonl(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk CAP Endometrium v5.1 into reference JSONL")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"CAP PDF not found at {args.pdf}")

    # raw/cleaned text are just scratch files for this run, no need to keep them around
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        text = extract_pdf_text(args.pdf)
        print(f"Extracted {len(text)} chars from {args.pdf.name}")
        (tmp_dir / "raw.txt").write_text(text, encoding="utf-8")

        cleaned = strip_page_footers(text)
        print(f"Stripped page footers: {len(text)} -> {len(cleaned)} chars")
        (tmp_dir / "clean.txt").write_text(cleaned, encoding="utf-8")

        sections = extract_explanatory_notes(cleaned)
        print(f"Found {len(sections)} explanatory notes: {sorted(sections.keys())}")

        expected_letters = set(NOTE_TO_GRAPH_NODES.keys())
        found_letters = set(sections.keys())
        if found_letters != expected_letters:
            raise ValueError(
                f"Section extraction mismatch: expected {sorted(expected_letters)}, "
                f"got {sorted(found_letters)}"
            )
        for letter, data in sections.items():
            print(f"  {letter}. {data['title']} ({len(data['body'])} chars)")

        mapped_letters = {letter for letter, nodes in NOTE_TO_GRAPH_NODES.items() if nodes}
        unmapped_letters = set(sections.keys()) - mapped_letters

        mapped_sections = {l: sections[l] for l in mapped_letters if l in sections}
        unmapped_sections = {l: sections[l] for l in unmapped_letters if l in sections}

        mapped_chunks = sections_to_chunks(mapped_sections, NOTE_TO_GRAPH_NODES)
        unmapped_chunks = sections_to_chunks(unmapped_sections, NOTE_TO_GRAPH_NODES)

        write_jsonl(mapped_chunks, OUTPUT_DIR / "mapped" / "mapped.jsonl")
        write_jsonl(unmapped_chunks, OUTPUT_DIR / "unmapped" / "unmapped.jsonl")
    # tmp dir and its raw/clean text are deleted here, only the jsonl files remain

    print(f"Wrote {len(mapped_chunks)} mapped and {len(unmapped_chunks)} unmapped chunks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
