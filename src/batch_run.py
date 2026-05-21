from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime

from pdf_reader import load_document
from preprocess import normalize_whitespace, clean_pdf_noise, split_into_sentences
from extractive import extractive_summary_mmr
from llm_summarizer import llm_summary_ollama
from evaluator import build_comparison_report


OUTPUT_DIR = Path("data/output")


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", errors="ignore")


def main():
    input_dir = Path("data/input")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.txt"))
    if not files:
        print("ERROR: No input files found in data/input/")
        return

    print(f"Found {len(files)} files. Running batch experiment...\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = OUTPUT_DIR / f"batch_{stamp}"
    batch_folder.mkdir(parents=True, exist_ok=True)

    for f in files:
        print("====================================")
        print("File:", f.name)

        raw_text = load_document(f)
        clean_text = clean_pdf_noise(raw_text) if f.suffix.lower() == ".pdf" else normalize_whitespace(raw_text)

        sentences = split_into_sentences(clean_text)
        if len(sentences) > 180:
            sentences = sentences[:180]

        t0 = time.time()
        extractive, _ = extractive_summary_mmr(sentences, max_sentences=8)
        extractive_time = time.time() - t0

        t1 = time.time()
        llm_sum = llm_summary_ollama(clean_text, model="llama3.2:1b")
        llm_time = time.time() - t1

        compare_text, eval_res = build_comparison_report(
            filename=f.name,
            clean_text=clean_text,
            extractive_summary=extractive,
            llm_summary=llm_sum,
            extractive_time_s=extractive_time,
            llm_time_s=llm_time,
        )

        out_compare = batch_folder / f"{f.stem}_comparison.txt"
        save_text(out_compare, compare_text)

        print(f"Extractive: {extractive_time:.2f}s | LLM: {llm_time:.2f}s | Hallucination flag: {eval_res.hallucination_flag}")
        print("Saved:", out_compare.name)

    print("\nBatch finished.")
    print("Results folder:", batch_folder.resolve())


if __name__ == "__main__":
    main()
