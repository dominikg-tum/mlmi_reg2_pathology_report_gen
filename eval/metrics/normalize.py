import re


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"(?<!\d)[^\w\s](?!\d)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text