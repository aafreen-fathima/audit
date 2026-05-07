# PRD — Audit

A small, honest LLM eval pipeline. Compares GPT-4o, Claude Sonnet, and Claude Haiku on a real product task — the user-interview extraction from Project 2 — and reports precision, recall, and quote attribution accuracy.

**Author:** [Your Name]
**Status:** v0.1 — built as portfolio project
**Last updated:** May 2026

---

## 1. The problem

Most teams shipping LLM features make model and prompt decisions on vibes:

- "GPT-4o feels better, let's switch."
- "We tried Claude on a few examples and it seemed fine."
- "We changed the prompt and it works better now."

This is how you ship regressions. Without evals, every model swap, prompt tweak, or vendor change is a coin flip on whether your product just got worse.

The opportunity: a tiny, opinionated eval harness that's cheap to run, fast to extend, and surfaces the actual numbers in a dashboard instead of buried in CI logs.

## 2. Target user

**Primary:** AI engineers and PMs at startups shipping LLM features who *know* they should have evals but haven't found the time to set them up properly.

**Secondary:** Me — running this on Project 2 to make defensible model choices.

**Not the target:** Academic NLP benchmarking. Production-scale eval ops (Arize, Galileo, Langfuse exist for that).

## 3. Goals & non-goals

**Goals**
- Compare ≥3 models (GPT-4o, Claude Sonnet, Claude Haiku) on one product task.
- Report precision, recall, F1, and quote attribution accuracy.
- Run end-to-end in under 5 minutes.
- Output: a JSON results file + a Streamlit dashboard with per-question drilldown.

**Non-goals (v0.1)**
- LLM-as-judge evals (next iteration).
- Latency / cost dashboards (mentioned but not first-class).
- Multi-task. v0.1 evaluates *one* task well.

## 4. The task being evaluated

User-interview pain-point extraction (Project 2 step 1). For each transcript, the model must return:

```json
{
  "extractions": [
    {"claim": "...", "verbatim_quote": "...", "severity": 1-5, "category": "..."}
  ]
}
```

The eval set has 5 transcripts, each labeled by hand with ~5 ground-truth pain points (so ~25 reference items total). Ground truth is `(claim, category)` — we don't pin exact wording because paraphrasing is fine.

## 5. The metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| **Recall** | % of ground-truth pain points the model caught (matched by semantic similarity ≥ 0.8) | "Does the model miss things?" — the most expensive failure mode for this product. |
| **Precision** | % of model extractions that match a ground-truth item | "Does the model invent things?" — the second-most expensive failure mode. |
| **F1** | Harmonic mean | Single number for ranking models. |
| **Quote attribution accuracy** | % of `verbatim_quote` fields that are real substrings of the transcript | Catches hallucinated quotes — the failure that destroys user trust fastest. |
| **Mean cost / transcript** | Tokens × price | Tiebreaker between models with similar quality. |

A semantic match (cosine ≥ 0.8 between embedded claims) is used to align extractions to ground truth. This is itself a noisy step — but it's *consistently* noisy across models, which is what matters for ranking.

## 6. The matchup

| Model | Why it's in the eval |
|---|---|
| Claude Sonnet 4.6 | Default for Project 2. Highest quality bar, highest cost. |
| Claude Haiku 4.5 | 5x cheaper than Sonnet. Question: is the quality drop worth it? |
| GPT-4o | The other major frontier model. Different lineage, different failure modes. |

## 7. Success metrics for *this product*

This project itself succeeds if:

- I can defensibly answer "which model should Project 2 use?" with numbers, not vibes.
- A reader can clone the repo, run `python run_evals.py`, and see results in 5 minutes.
- The dashboard makes it obvious which model wins on which metric — not just an aggregate.

## 8. Risks & limitations

| Risk | Mitigation |
|---|---|
| Eval set is too small (5 transcripts) — high variance | Honestly noted in the dashboard. v0.2 expands to 30+. |
| Ground truth is one PM's opinion | Add second annotator in v0.2. |
| Semantic matching threshold (0.8) is arbitrary | Threshold sensitivity sweep in `notebooks/`. |
| LLM outputs are non-deterministic | Run each model 3x, report mean and stddev. |

## 9. Roadmap

**v0.1 (shipped):** 5-transcript eval set, 3 models, 4 metrics, dashboard.
**v0.2:** 30-transcript eval set with second annotator. LLM-as-judge metric for "is this extraction actionable for a PM?" (different from precision/recall).
**v0.3:** CI integration — block merges that drop F1 by >2%.
**v1.0:** Multi-task. Plug in a task definition, get a model comparison.

## 10. What I'd ship next

A "regression mode" that runs the evals against the *prompt's* git history. Most teams have prompts in code; very few teams know which prompt change made things better or worse. Showing F1 over the last 30 days of prompt commits would be a small thing with outsized impact on AI eng workflow. That's the kind of internal-tool wedge I'd push for if this were a real product.
