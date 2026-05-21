from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict


@dataclass
class EvalResult:
    doc_chars: int
    extractive_chars: int
    llm_chars: int
    extractive_time_s: float
    llm_time_s: float
    hallucination_flag: bool
    hallucination_notes: str


def _extract_numbers(text: str) -> set[str]:
    # captures things like 100, 1.12, 3.0, 20 months, 45 KLOC (basic)
    nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    return nums


def hallucination_check(clean_text: str, llm_text: str) -> tuple[bool, str]:
    """
    Simple, explainable check:
    - Extract numbers from document and from LLM output
    - If LLM introduces many numbers not in doc -> flag
    NOTE: This is not perfect, but it's great for ICA evaluation discussion.
    """
    doc_nums = _extract_numbers(clean_text)
    llm_nums = _extract_numbers(llm_text)

    extra = sorted(list(llm_nums - doc_nums))
    if len(extra) == 0:
        return False, "No extra numbers detected."

    # If LLM adds more than 2 new numbers, treat as suspicious
    if len(extra) > 2:
        return True, f"LLM introduced numbers not in document: {', '.join(extra[:20])}" + (" ..." if len(extra) > 20 else "")

    return False, f"Minor extra numbers detected: {', '.join(extra)}"


def build_comparison_report(
    filename: str,
    clean_text: str,
    extractive_summary: str,
    llm_summary: str,
    extractive_time_s: float,
    llm_time_s: float,
) -> tuple[str, EvalResult]:
    flag, notes = hallucination_check(clean_text, llm_summary)

    report = []
    report.append("=== DOCUMENT SUMMARIZATION COMPARISON REPORT ===")
    report.append(f"File: {filename}")
    report.append("")
    report.append("== RUNTIME ==")
    report.append(f"Extractive time (s): {extractive_time_s:.2f}")
    report.append(f"LLM time (s):        {llm_time_s:.2f}")
    report.append("")
    report.append("== LENGTHS ==")
    report.append(f"Document chars:      {len(clean_text)}")
    report.append(f"Extractive chars:    {len(extractive_summary)}")
    report.append(f"LLM chars:           {len(llm_summary)}")
    report.append("")
    report.append("== HALLUCINATION CHECK (numbers) ==")
    report.append(f"Flag: {flag}")
    report.append(f"Notes: {notes}")
    report.append("")
    report.append("== EXTRACTIVE SUMMARY (TF-IDF + MMR) ==")
    report.append(extractive_summary.strip() or "[EMPTY]")
    report.append("")
    report.append("== LLM SUMMARY (Ollama) ==")
    report.append(llm_summary.strip() or "[EMPTY]")
    report.append("")
    report.append("== EVALUATION NOTES TEMPLATE (for your report) ==")
    report.append("- Accuracy: Did the summary match the PDF content?")
    report.append("- Coverage: Did it include all main tasks/points?")
    report.append("- Conciseness: Is it short but complete?")
    report.append("- Factual reliability: Any invented numbers/claims?")
    report.append("- Practicality: Time taken + offline privacy benefits.")
    report.append("")

    eval_res = EvalResult(
        doc_chars=len(clean_text),
        extractive_chars=len(extractive_summary),
        llm_chars=len(llm_summary),
        extractive_time_s=extractive_time_s,
        llm_time_s=llm_time_s,
        hallucination_flag=flag,
        hallucination_notes=notes,
    )
    return "\n".join(report), eval_res
