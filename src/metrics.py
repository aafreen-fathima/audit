"""Metrics: precision, recall, F1, quote attribution accuracy.

Semantic alignment uses cosine similarity on embeddings ≥ 0.8.
The threshold is empirically chosen — see PRD section 8."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.8


@dataclass
class CaseMetrics:
    case_id: str
    model: str
    run_idx: int
    precision: float
    recall: float
    f1: float
    quote_attribution: float
    n_extracted: int
    n_truth: int
    n_matched: int
    cost_usd: float


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def quote_attribution_accuracy(extractions: list[dict], transcript: str) -> float:
    """% of verbatim_quotes that appear as substrings of the transcript."""
    if not extractions:
        return 1.0  # vacuously true; no claims, no false attributions
    norm_transcript = _normalize(transcript)
    hits = sum(1 for e in extractions if _normalize(e.get("verbatim_quote", "")) in norm_transcript)
    return hits / len(extractions)


def _embed(texts: list[str], openai_client: OpenAI) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536))
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def precision_recall(
    extracted_claims: list[str],
    ground_truth_claims: list[str],
    openai_client: OpenAI,
    threshold: float = MATCH_THRESHOLD,
) -> tuple[float, float, int]:
    """Greedy 1:1 matching by cosine similarity. Returns (precision, recall, n_matched)."""
    if not extracted_claims and not ground_truth_claims:
        return 1.0, 1.0, 0
    if not extracted_claims:
        return 0.0, 0.0, 0
    if not ground_truth_claims:
        return 0.0, 1.0, 0

    extracted_emb = _embed(extracted_claims, openai_client)
    truth_emb = _embed(ground_truth_claims, openai_client)
    sim = _cosine_matrix(extracted_emb, truth_emb)

    # Greedy bipartite matching: pick highest-similarity pair, remove row/col, repeat.
    matched_truth: set[int] = set()
    matched_extracted: set[int] = set()
    flat = [(sim[i, j], i, j) for i in range(sim.shape[0]) for j in range(sim.shape[1])]
    flat.sort(reverse=True)
    for s, i, j in flat:
        if s < threshold:
            break
        if i in matched_extracted or j in matched_truth:
            continue
        matched_extracted.add(i)
        matched_truth.add(j)

    n_matched = len(matched_extracted)
    precision = n_matched / len(extracted_claims) if extracted_claims else 0.0
    recall = n_matched / len(ground_truth_claims) if ground_truth_claims else 0.0
    return precision, recall, n_matched


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
