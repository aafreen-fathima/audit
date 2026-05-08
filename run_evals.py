"""Run the eval suite: every transcript × every model × N runs.

Writes results/latest.json. The dashboard reads from that file.

Usage:
    python run_evals.py [--runs N] [--models claude-sonnet-4-6,gpt-4o]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from src.metrics import CaseMetrics, f1, precision_recall, quote_attribution_accuracy
from src.models import MODELS
from src.tasks import load_eval_set

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"


def run_one(
    case,
    model_name: str,
    run_idx: int,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> tuple[CaseMetrics, dict]:
    """Run a single (case, model, run) cell. Returns metrics + raw output for debugging."""
    extract_fn = MODELS[model_name]
    # Anthropic and OpenAI extracts have different first args
    if model_name.startswith("claude"):
        out = extract_fn(anthropic_client, case.transcript)
    else:
        out = extract_fn(openai_client, case.transcript)

    extracted = out.extractions
    extracted_claims = [e.get("claim", "") for e in extracted]
    truth_claims = [g.claim for g in case.ground_truth]

    p, r, n_matched = precision_recall(extracted_claims, truth_claims, openai_client)
    f1_score = f1(p, r)
    qa = quote_attribution_accuracy(extracted, case.transcript)

    metrics = CaseMetrics(
        case_id=case.id,
        model=model_name,
        run_idx=run_idx,
        precision=p,
        recall=r,
        f1=f1_score,
        quote_attribution=qa,
        n_extracted=len(extracted),
        n_truth=len(case.ground_truth),
        n_matched=n_matched,
        cost_usd=out.cost_estimate_usd,
    )
    return metrics, {"extracted": extracted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per (case, model).")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help="Comma-separated model IDs.",
    )
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not openai_key or not anthropic_key:
        raise SystemExit("Set OPENAI_API_KEY and ANTHROPIC_API_KEY in your .env")

    openai_client = OpenAI(api_key=openai_key)
    anthropic_client = Anthropic(api_key=anthropic_key)

    cases = load_eval_set()
    models = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    print(f"Eval set: {len(cases)} cases. Models: {models}. Runs each: {args.runs}.")
    print(f"Total cells: {len(cases) * len(models) * args.runs}")
    print()

    all_metrics: list[CaseMetrics] = []
    raw_outputs: list[dict] = []
    t0 = time.time()

    for case in cases:
        for model_name in models:
            for run_idx in range(args.runs):
                print(f"  • {case.id} × {model_name} × run {run_idx+1}/{args.runs}", end=" ", flush=True)
                try:
                    metrics, raw = run_one(case, model_name, run_idx, anthropic_client, openai_client)
                    all_metrics.append(metrics)
                    raw_outputs.append({
                        "case_id": case.id,
                        "model": model_name,
                        "run_idx": run_idx,
                        **raw,
                    })
                    print(f"F1={metrics.f1:.2f} R={metrics.recall:.2f} P={metrics.precision:.2f} QA={metrics.quote_attribution:.2f}")
                except Exception as exc:
                    print(f"ERROR: {exc}")

    elapsed = time.time() - t0
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "latest.json"
    out_path.write_text(json.dumps({
        "elapsed_s": elapsed,
        "metrics": [asdict(m) for m in all_metrics],
        "raw_outputs": raw_outputs,
    }, indent=2))
    print()
    print(f"Wrote {out_path} ({elapsed:.1f}s elapsed). View dashboard: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
