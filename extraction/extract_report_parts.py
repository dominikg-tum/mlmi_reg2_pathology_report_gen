"""WP3: extract organ / specimen / procedure from reports via local Qwen (vLLM)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]

SLIDE_ID_COLUMN = "slide_ids"
REPORT_COLUMN = "english_reports"
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 2

SYSTEM_PROMPT = """
You are an expert pathology information extraction system.

Extract exactly these fields:
1. organ
2. type_of_specimen
3. procedure

Rules:
- Return valid JSON only.
- Do not explain.
- Do not use markdown.
- If information is missing, unclear, or not relevant, return "N.A."
- If multiple values exist, separate them with "; ".
- Keep values concise and standardized.
"""


def load_config() -> dict[str, Any]:
    import yaml

    path = REPO_ROOT / "configs" / "paths.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def clean_json_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return text.strip()


def normalize_value(value: Any) -> str:
    if value is None:
        return "N.A."
    value = str(value).strip()
    if value == "":
        return "N.A."
    if value.lower() in ["na", "n/a", "none", "null", "not available", "not applicable"]:
        return "N.A."
    return value


def extract_parts(client: OpenAI, model: str, report_text: str) -> dict:
    user_prompt = f"""
Extract the following information from this pathology report.

Fields:

organ:
Anatomical location or organ.

type_of_specimen:
Material submitted to pathology.

procedure:
Histological, cytological, molecular, or laboratory procedure.

Return exactly this JSON structure:

{{
  "organ": "...",
  "type_of_specimen": "...",
  "procedure": "..."
}}

Pathology report:
\"\"\"
{report_text}
\"\"\"
"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            content = clean_json_response(response.choices[0].message.content or "")
            parsed = json.loads(content)
            return {
                "organ": normalize_value(parsed.get("organ")),
                "type_of_specimen": normalize_value(parsed.get("type_of_specimen")),
                "procedure": normalize_value(parsed.get("procedure")),
            }
        except Exception as e:
            if attempt == MAX_RETRIES:
                return {
                    "organ": "N.A.",
                    "type_of_specimen": "N.A.",
                    "procedure": "N.A.",
                    "error": str(e),
                }
            time.sleep(SLEEP_BETWEEN_RETRIES)

    return {"organ": "N.A.", "type_of_specimen": "N.A.", "procedure": "N.A."}


def main() -> None:
    cfg = load_config()
    input_xlsx = Path(cfg["cluster"]["labels_xlsx"])
    extraction_cfg = cfg.get("extraction") or {}
    output_raw = extraction_cfg.get("report_parts_json")
    output_json = Path(output_raw) if output_raw else REPO_ROOT / "data" / "report_parts_extracted.json"
    qwen = cfg["qwen"]
    client = OpenAI(base_url=qwen["api_base_url"], api_key=qwen["api_key"])
    model = qwen["model_name"]

    if not input_xlsx.exists():
        raise FileNotFoundError(f"Input file not found: {input_xlsx}")

    df = pd.read_excel(input_xlsx)
    if SLIDE_ID_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {SLIDE_ID_COLUMN}")
    if REPORT_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {REPORT_COLUMN}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        slide_id = normalize_value(row[SLIDE_ID_COLUMN])
        report = row[REPORT_COLUMN]
        if pd.isna(report) or str(report).strip() == "":
            extracted = {
                "organ": "N.A.",
                "type_of_specimen": "N.A.",
                "procedure": "N.A.",
            }
        else:
            extracted = extract_parts(client, model, str(report))

        entry = {
            "slide_id": slide_id,
            "organ": extracted.get("organ", "N.A."),
            "type_of_specimen": extracted.get("type_of_specimen", "N.A."),
            "procedure": extracted.get("procedure", "N.A."),
        }
        if "error" in extracted:
            entry["error"] = extracted["error"]
        results.append(entry)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} entries to {output_json}")


if __name__ == "__main__":
    main()
