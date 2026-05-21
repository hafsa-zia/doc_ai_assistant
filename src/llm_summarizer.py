from __future__ import annotations
import subprocess
from typing import List


def ollama_available() -> bool:
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return True
    except Exception:
        return False


def _ollama_run(prompt: str, model: str) -> str:
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return f"[LLM ERROR] {proc.stderr.strip() or 'Unknown error calling ollama.'}"
    return (proc.stdout or "").strip()


def chunk_text(text: str, chunk_chars: int = 2500, overlap: int = 150, max_chunks: int = 4) -> List[str]:
    text = text.strip()
    chunks = []
    start = 0
    n = len(text)

    while start < n and len(chunks) < max_chunks:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def llm_summary_ollama(text: str, model: str = "llama3.2:1b") -> str:
    """
    LLM summary with anti-hallucination rules:
    - Do NOT invent numbers or results
    - If a number/formula isn't explicitly in the document: say "Not provided"
    """
    if not ollama_available():
        return "[LLM SUMMARY SKIPPED] Ollama not installed or not found in PATH."

    RULES = (
        "RULES (must follow strictly):\n"
        "1) Use ONLY information explicitly present in the document.\n"
        "2) Do NOT invent numbers, calculations, formulas, names, or results.\n"
        "3) If the document does not provide a value, write: 'Not provided in the document'.\n"
        "4) If uncertain, say: 'Unclear from the document'.\n"
        "5) Keep bullet points short and factual.\n"
        "6)Never output filler text like 'Note provided...'. If uncertain, say 'Not provided in the document'.\n"

    )

    # short doc: one pass
    if len(text) <= 3500:
        prompt = (
            f"{RULES}\n"
            "Task: Summarize the document.\n"
            "Output format:\n"
            "- 6 to 10 bullet points\n"
            "- then a short paragraph (5–8 lines)\n\n"
            f"DOCUMENT:\n{text}\n"
        )
        return _ollama_run(prompt, model=model)

    # longer doc: chunk + merge
    chunks = chunk_text(text, chunk_chars=2500, overlap=150, max_chunks=4)
    chunk_summaries = []

    for i, ch in enumerate(chunks, start=1):
        print(f"  [LLM] Summarizing chunk {i}/{len(chunks)} ...")
        prompt = (
            f"{RULES}\n"
            "Task: Summarize this chunk in 4–6 bullet points.\n\n"
            f"CHUNK:\n{ch}\n"
        )
        chunk_summaries.append(_ollama_run(prompt, model=model))

    print("  [LLM] Merging chunk summaries ...")
    merge_prompt = (
        f"{RULES}\n"
        "Task: Combine the chunk summaries into ONE final summary.\n"
        "Output format:\n"
        "- 8 to 12 bullet points\n"
        "- then a short paragraph (5–8 lines)\n\n"
        "CHUNK SUMMARIES:\n" + "\n\n".join(chunk_summaries)
    )

    return _ollama_run(merge_prompt, model=model)
