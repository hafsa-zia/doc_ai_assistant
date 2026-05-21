from __future__ import annotations

import csv
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


def append_csv_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    input_dir = Path("data/input")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Intelligent Document Assistant  ===")
    print(f"Input folder:  {input_dir.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    supported = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.txt"))
    if not supported:
        print("ERROR: No .pdf or .txt found in data/input/")
        return

    file_path = supported[0]
    print(f"\nUsing file: {file_path.name}")

    raw_text = load_document(file_path)

    if file_path.suffix.lower() == ".pdf":
        clean_text = clean_pdf_noise(raw_text)
    else:
        clean_text = normalize_whitespace(raw_text)

    sentences = split_into_sentences(clean_text)

    print(f"\nCharacters (cleaned): {len(clean_text)}")
    print(f"Sentences (filtered):  {len(sentences)}")

    # Speed safeguard for extractive only
    MAX_SENTENCES = 180
    if len(sentences) > MAX_SENTENCES:
        sentences = sentences[:MAX_SENTENCES]
        print(f"Sentences capped to:   {len(sentences)} (speed safeguard)")

    # Extractive summary
    t0 = time.time()
    print("\nRunning extractive summarizer...")
    extractive, chosen_idx = extractive_summary_mmr(sentences, max_sentences=8)
    extractive_time = time.time() - t0
    print(f"Extractive done in {extractive_time:.2f}s")

    # LLM summary
    t1 = time.time()
    print("\nRunning LLM summarizer (Ollama)...")
    llm_sum = llm_summary_ollama(clean_text, model="llama3.2:1b")
    llm_time = time.time() - t1
    print(f"LLM done in {llm_time:.2f}s")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = file_path.stem

    out_full = OUTPUT_DIR / f"{base}_cleantext_{stamp}.txt"
    out_extractive = OUTPUT_DIR / f"{base}_extractive_mmr_{stamp}.txt"
    out_llm = OUTPUT_DIR / f"{base}_llm_{stamp}.txt"
    out_compare = OUTPUT_DIR / f"{base}_comparison_{stamp}.txt"
    out_csv = OUTPUT_DIR / "experiment_results.csv"

    save_text(out_full, clean_text)
    save_text(out_extractive, extractive)
    save_text(out_llm, llm_sum)

    comparison_text, eval_res = build_comparison_report(
        filename=file_path.name,
        clean_text=clean_text,
        extractive_summary=extractive,
        llm_summary=llm_sum,
        extractive_time_s=extractive_time,
        llm_time_s=llm_time,
    )
    save_text(out_compare, comparison_text)

    append_csv_row(out_csv, {
        "file": file_path.name,
        "doc_chars": eval_res.doc_chars,
        "extractive_chars": eval_res.extractive_chars,
        "llm_chars": eval_res.llm_chars,
        "extractive_time_s": f"{eval_res.extractive_time_s:.2f}",
        "llm_time_s": f"{eval_res.llm_time_s:.2f}",
        "hallucination_flag": str(eval_res.hallucination_flag),
        "hallucination_notes": eval_res.hallucination_notes,
        "timestamp": stamp,
    })

    print("\nSaved outputs:")
    print(" -", out_full.name)
    print(" -", out_extractive.name)
    print(" -", out_llm.name)
    print(" -", out_compare.name)
    print(" - experiment_results.csv")

    print("\n=== Extractive Summary (MMR TF-IDF) ===\n")
    print(extractive)

    print("\n=== LLM Summary (Ollama) ===\n")
    print(llm_sum)

    print("\n=== Hallucination check ===")
    print("Flag:", eval_res.hallucination_flag)
    print("Notes:", eval_res.hallucination_notes)


if __name__ == "__main__":
    main()
