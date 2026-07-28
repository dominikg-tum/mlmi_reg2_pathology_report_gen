import re


def normalize_answer(text: str) -> str:
    text = text.lower().strip()

    def _replace_punct(m: re.Match) -> str:
        start, end = m.start(), m.end()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        # punctuation stuck between two letters would merge them into one
        # token if just dropped ("endometrioid/serous" -> "endometrioidserous")
        if before.isalpha() and after.isalpha():
            return " "
        return ""

    text = re.sub(r"(?<!\d)[^\w\s](?!\d)", _replace_punct, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text