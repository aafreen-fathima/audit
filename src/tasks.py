"""Eval set loader. Self-contained — does not import Project 2."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EVAL_SET_PATH = Path(__file__).parent / "data" / "interview_eval_set.json"


@dataclass
class GroundTruthItem:
    claim: str
    category: str


@dataclass
class EvalCase:
    id: str
    transcript: str
    ground_truth: list[GroundTruthItem]


def load_eval_set() -> list[EvalCase]:
    """Load eval set + transcript text. Resolves transcript paths relative to JSON."""
    with EVAL_SET_PATH.open() as f:
        data = json.load(f)

    cases: list[EvalCase] = []
    for tx in data["transcripts"]:
        transcript_path = (EVAL_SET_PATH.parent / tx["path"]).resolve()
        try:
            transcript = transcript_path.read_text()
        except FileNotFoundError:
            # Allow standalone use of the eval project — fall back to a stub
            transcript = f"[Transcript not found at {transcript_path}. Run from the portfolio repo.]"
        cases.append(EvalCase(
            id=tx["id"],
            transcript=transcript,
            ground_truth=[GroundTruthItem(**g) for g in tx["ground_truth"]],
        ))
    return cases
