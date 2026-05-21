from __future__ import annotations
import re
from typing import List


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pdf_noise(text: str) -> str:
    """
    Strong PDF cleaner:
    - remove page numbers
    - remove Figure/Table captions
    - remove dot leaders "....."
    - remove symbol-only lines
    - remove common assignment/admin lines (instructions, due date, submission, etc.)
    """
    text = normalize_whitespace(text)
    lines = [ln.strip() for ln in text.split("\n")]

    cleaned = []
    for ln in lines:
        if not ln:
            continue

        low = ln.lower()

        # page numbers alone
        if re.fullmatch(r"\d{1,4}", ln):
            continue

        # figure/table captions
        if re.match(r"^(figure|table)\s*\d+", ln, flags=re.IGNORECASE):
            continue

        # dot leader lines
        if re.search(r"\.{8,}", ln):
            continue

        # symbol-only lines
        if re.fullmatch(r"[-_•=]{3,}", ln):
            continue

        # common admin/instruction noise (helps summaries a LOT)
        admin_keywords = [
            "instructions", "due date", "submission", "submitted", "hand written",
            "google classroom", "originality", "timely submission", "max", "members",
            "task #", "task#", "assignment should", "scan copy"
        ]
        if any(k in low for k in admin_keywords):
            continue

        # remove tiny header-like tokens (e.g., repeated title)
        if len(ln) <= 14 and re.fullmatch(r"[A-Za-z0-9]+", ln):
            continue
# remove emails / contact lines
        if re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", ln):
            continue

# remove reference/citation-only lines like [12] ...
        if re.match(r"^\[\d+\]\s", ln):
            continue
        cleaned.append(ln)

    return "\n".join(cleaned)


def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)

    # protect abbreviations
    text = re.sub(
        r"\b(e\.g|i\.e|mr|mrs|dr|prof)\.",
        lambda m: m.group(0).replace(".", "<DOT>"),
        text,
        flags=re.IGNORECASE,
    )

    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.replace("<DOT>", ".").strip() for s in sents]

    filtered: List[str] = []
    for s in sents:
        s = s.strip()
        if len(s) < 35:
            continue

        # remove caption-like
        if re.match(r"^(figure|table)\s*\d+", s, flags=re.IGNORECASE):
            continue

        # dot leader artifacts
        if re.search(r"\.{8,}", s):
            continue

        # numbering-only artifacts like "10." or "3"
        if re.fullmatch(r"\d+\.?", s):
            continue

        # too little alphabetic content (junk)
        alpha_ratio = sum(ch.isalpha() for ch in s) / max(1, len(s))
        if alpha_ratio < 0.20:
            continue

        filtered.append(s)

    return filtered
