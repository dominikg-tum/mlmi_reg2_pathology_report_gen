'''
enroot start --root --rw \
  --mount /mnt:/mnt \
  --mount /tmp:/tmp \
  qwen25
'''

import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# =========================
# Config
# =========================

INPUT_XLSX = "/mnt/projects/mlmi/reg2/case_reports_to_korea_collaborators.xlsx"
OUTPUT_JSON = "/mnt/projects/mlmi/reg2/report_parts_extracted.json"

BASE_URL = "http://localhost:8000/v1"
API_KEY = "dummy"

MODEL_NAME = "/mnt/projects/mlmi/reg2/models/Qwen3-VL-8B-Instruct"

SLIDE_ID_COLUMN = "slide_ids"
REPORT_COLUMN = "english_reports"

MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 2


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


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


def clean_json_response(text: str) -> str:
    text = text.strip()

    # Remove markdown fences if the model adds them
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Extract first JSON object if extra text exists
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    return text.strip()


def normalize_value(value):
    if value is None:
        return "N.A."

    value = str(value).strip()

    if value == "":
        return "N.A."

    if value.lower() in ["na", "n/a", "none", "null", "not available", "not applicable"]:
        return "N.A."

    return value


def extract_parts(report_text: str) -> dict:
    user_prompt = f"""
Extract the following information from this pathology report.

Fields:

organ:
Anatomical location or organ.
Examples: cervix; uterus; endometrium; ovary; fallopian tube; vulva; vagina.

type_of_specimen:
Material submitted to pathology.
Examples: biopsy; curettage; resection specimen; cytology specimen; liquid material; cell block.

procedure:
Histological, cytological, molecular, or laboratory procedure.
Examples: HE staining; PAS staining; Papanicolaou staining; HEmacolor staining; immunohistochemistry; p53; MLH1; PMS2.

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
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )

            content = response.choices[0].message.content
            content = clean_json_response(content)

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

    return {
        "organ": "N.A.",
        "type_of_specimen": "N.A.",
        "procedure": "N.A.",
    }


def main():
    input_path = Path(INPUT_XLSX)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_XLSX}")

    df = pd.read_excel(INPUT_XLSX)

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
            extracted = extract_parts(str(report))

        entry = {
            "slide_id": slide_id,
            "organ": extracted.get("organ", "N.A."),
            "type_of_specimen": extracted.get("type_of_specimen", "N.A."),
            "procedure": extracted.get("procedure", "N.A."),
        }

        if "error" in extracted:
            entry["error"] = extracted["error"]

        results.append(entry)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} entries to:")
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()