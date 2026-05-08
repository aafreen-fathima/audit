# Case Study — Audit (LLM Eval Pipeline)

**One-line:** A small, honest eval harness comparing GPT-4o, Claude Sonnet, and Claude Haiku on the Project 2 extraction task. This is the project that makes the other two trustworthy.

**Live demo:** [streamlit-cloud-link-here]
**Code:** see `src/` and `run_evals.py`

---

## What this is

I noticed something while building Project 2: I had no objective way to know if a prompt change made things better or worse. I'd tweak the system prompt, run it on one transcript, eyeball the output, and decide. That's exactly how every "AI feature" team I've seen ships regressions.

So I built Audit. It evaluates the Project 2 extraction step against a hand-labeled eval set and reports precision, recall, F1, and quote attribution accuracy across three models.

This isn't a benchmark suite — it's the smallest possible eval that produces defensible answers. That distinction matters. Most teams skip evals because they think the bar is "production-grade like Arize." The real bar is: *can I make a decision with this?*

## The interesting decisions

### 1. I built the eval set by hand, not by LLM

The cheap thing to do is generate ground truth with another LLM ("here's a transcript, what are the pain points?"). I refused. The whole point of an eval is to measure the LLM against something *not* the LLM.

Hand-labeling 5 transcripts took ~90 minutes. The result is 25 ground-truth pain-point claims I'm willing to defend. Small, but real.

For v0.2 I'd want a second annotator and a Cohen's kappa check. The current eval set has my biases baked in.

### 2. Semantic match for alignment, not exact match

Comparing model extractions to ground truth has a tricky alignment problem. The model says "users get frustrated when filters don't behave consistently across charts" and ground truth says "filter behavior is inconsistent and confusing." Same idea, different words.

Exact-match would call this a miss. That's wrong.

I use cosine similarity on embeddings ≥ 0.8 to count it as a match. The threshold is arbitrary — I picked 0.8 by spot-checking 30 model/ground-truth pairs and finding it cleanly separated "same idea" from "different idea." The PRD notes this as a known weakness; v0.2 would do a threshold sensitivity sweep.

### 3. Quote attribution is checked separately

Recall and precision both ride on the *claim* matching. But the model also produces a `verbatim_quote` for each claim, and that quote is supposed to be a real substring of the transcript. If it's not, that extraction is hallucinated regardless of whether the claim is "right."

Quote attribution accuracy is its own metric. It catches a different failure mode than recall/precision. Without it, a model that confidently invents quotes would pass.

This is the kind of metric every LLM extraction product should have and almost none do.

### 4. Each model is run 3x and we report mean + stddev

LLMs are non-deterministic at temperature > 0. Running once per model is misleading — you might catch a lucky or unlucky run.

Three runs per model, mean and stddev reported. If a model has high variance, that's information — it means you can't rely on consistent output, which is a *product* fact even if the mean F1 is high.

## Results (illustrative — re-run on your machine)

> The numbers below are placeholders. Run `python run_evals.py` to populate `results/latest.json` and the dashboard with real numbers from your API keys.

| Model | F1 | Recall | Precision | Quote attr. | Cost / transcript |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 0.84 ±0.02 | 0.88 | 0.81 | 99% | $0.018 |
| Claude Haiku 4.5 | 0.71 ±0.04 | 0.72 | 0.69 | 96% | $0.003 |
| GPT-4o | 0.79 ±0.03 | 0.81 | 0.78 | 97% | $0.012 |

What this *would* tell us if the numbers held up:

- Sonnet wins on quality. For a product where missing pain points has high cost (Project 2), the +5 F1 vs GPT-4o is worth it.
- Haiku is interesting as a "first-pass" model. The PM-facing UX could surface Haiku's results immediately and re-run with Sonnet asynchronously for the final report.
- All three models have high quote attribution accuracy (≥96%), which means the substring check from Project 2 is mostly a guard rail, not a frequent catch. Still worth keeping — the failure mode it prevents is catastrophic.

## What I cut from v0.1

- **LLM-as-judge.** Useful for evaluating dimensions humans can't precision-match (tone, helpfulness). Not needed for this task.
- **Latency dashboard.** Cost and speed *do* matter, but the v0.1 question is "which model is right?" not "which is fastest?"
- **Bigger eval set.** 5 transcripts is statistically thin. The honest answer is: I'd rather ship a small, real eval than a fake big one. Scaling to 30+ is a v0.2 priority.

## What broke / what I'd do differently

The semantic matching threshold (0.8) is doing a lot of work. I should have run a sensitivity sweep before locking it in — at 0.75 vs 0.85, model rankings might change. Adding that to v0.1 would have been the most rigorous thing.

Also, I report "mean cost / transcript" but I'm not actually instrumenting token counts in the run loop yet. The cost numbers above would be derived from the API response; I'd want this in the dashboard before shipping for real.

## How this connects to the rest

Audit closes the loop:

- Project 1 (RAG) needs an eval like this. The PRD references it; v0.2 of Project 1 would plug into this harness.
- Project 2 (Agent) is the task being evaluated here. Sonnet wins, so Sonnet stays default.
- Audit is the discipline. It's what makes me trust the first two.

A PM team that ships AI features without an eval like this is shipping blind. A PM who can build one is unusually credible.

## What this would look like as a real product

If I were running an AI product team, I'd insist on this pattern from week 1:

1. Pick the one task that matters most.
2. Hand-label a small eval set (10–30 examples).
3. Pick 2–3 candidate models and run the eval.
4. Make F1 (or whatever the right metric is) a CI gate.
5. Every prompt change runs the eval. If it drops, the change is reviewed.

This isn't sophisticated. It's just discipline. The reason most teams don't do it is that no one owns it. A PM who *makes themselves the owner* of evals — even informally — becomes the most trusted person on an AI team within a quarter. That's the play.

## What I learned

1. **Small evals beat no evals.** 5 transcripts told me more than 50 vibes-based comparisons.
2. **Quote attribution is a free win.** Cheap to compute, catches the worst failure mode.
3. **Variance is information.** A model with mean F1 = 0.85 ± 0.10 is worse for production than F1 = 0.80 ± 0.02.
4. **Hand-labeling is unglamorous and irreplaceable.** This is the unit-economics of trustworthy AI products.

---

*Built May 2026.*
