from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extractive_summary_mmr(sentences: List[str], max_sentences: int = 8) -> Tuple[str, List[int]]:
    """
    Fast extractive summarization:
    - TF-IDF relevance scoring
    - reduce to top candidates
    - MMR for diversity on candidates (precomputed similarity matrix)
    """
    if not sentences:
        return "", []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=4000,
        min_df=2,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(sentences)
    doc_vec = np.asarray(X.mean(axis=0))
    sim_to_doc = cosine_similarity(X, doc_vec).ravel()

    # candidates reduction (prevents slowness and improves relevance)
    top_n = min(40, len(sentences))
    cand_idx = np.argsort(sim_to_doc)[::-1][:top_n].tolist()

    Xc = X[cand_idx]
    rel = sim_to_doc[cand_idx]
    S = cosine_similarity(Xc)  # small matrix (<=40x40)

    selected_local = []
    candidates_local = list(range(len(cand_idx)))
    diversity = 0.65

    first = int(np.argmax(rel))
    selected_local.append(first)
    candidates_local.remove(first)

    while len(selected_local) < min(max_sentences, len(cand_idx)) and candidates_local:
        best_score = -1e9
        best = None
        for c in candidates_local:
            redundancy = max(S[c, s] for s in selected_local)
            score = (1 - diversity) * rel[c] - diversity * redundancy
            if score > best_score:
                best_score = score
                best = c
        selected_local.append(best)
        candidates_local.remove(best)

    chosen_global = sorted(cand_idx[i] for i in selected_local)
    summary = "\n".join(sentences[i] for i in chosen_global)
    return summary, chosen_global
