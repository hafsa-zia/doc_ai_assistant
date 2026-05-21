from __future__ import annotations
from pathlib import Path
from typing import Optional
from pypdf import PdfReader


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


def load_document(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix in [".txt", ".text"]:
        return read_text_file(path)

    raise ValueError("Unsupported file type. Use .pdf or .txt")
