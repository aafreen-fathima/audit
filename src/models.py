"""Model wrappers with a unified extract() interface.

Each wrapper takes a transcript and returns list of {claim, verbatim_quote, severity, category}.
We re-implement the prompt + tool schema here so this project is self-contained
(no import dependency on Project 2)."""
from __future__ import annotations

from dataclasses import dataclass

from anthropic import Anthropic
from openai import OpenAI

# --- Prompts (kept in sync with Project 2's src/prompts.py) ---

EXTRACTION_SYSTEM = """You are a careful product researcher. Your job is to read user interview transcripts and extract distinct claims the user makes about pain, friction, or unmet need.

Strict rules:
1. Every claim MUST be supported by a verbatim quote — an exact substring of the transcript.
2. Paraphrase the claim itself. Do NOT paraphrase the quote.
3. One claim per distinct pain point.
4. Severity 1 = mild annoyance, 5 = blocks workflow.
5. Don't editorialize. Don't infer pain that isn't expressed.
6. Categories: onboarding, performance, pricing, integration, ux, reliability, feature_gap, other."""

EXTRACTION_USER_TEMPLATE = "Transcript:\n\n{transcript}\n\nExtract every distinct user pain or unmet need."


# --- Anthropic tool schema ---

EXTRACTION_TOOL = {
    "name": "record_extractions",
    "description": "Record extracted pain points.",
    "input_schema": {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "verbatim_quote": {"type": "string"},
                        "severity": {"type": "integer", "minimum": 1, "maximum": 5},
                        "category": {"type": "string"},
                    },
                    "required": ["claim", "verbatim_quote", "severity", "category"],
                },
            }
        },
        "required": ["extractions"],
    },
}


# --- OpenAI function-calling schema (equivalent shape) ---

OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "record_extractions",
        "description": "Record extracted pain points.",
        "parameters": EXTRACTION_TOOL["input_schema"],
    },
}


@dataclass
class ModelOutput:
    extractions: list[dict]
    raw_response: object  # for debugging
    cost_estimate_usd: float = 0.0


# --- Anthropic models ---

def _claude_extract(
    client: Anthropic,
    model_id: str,
    transcript: str,
    temperature: float = 0.3,
) -> ModelOutput:
    resp = client.messages.create(
        model=model_id,
        max_tokens=2048,
        temperature=temperature,
        system=EXTRACTION_SYSTEM,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_extractions"},
        messages=[{"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(transcript=transcript)}],
    )
    extractions: list[dict] = []
    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_extractions":
            extractions = block.input.get("extractions", [])
            break

    # Rough cost estimate. Actual Anthropic pricing varies — these are placeholders.
    # Update with current per-million-token rates from anthropic.com.
    rates = {
        "claude-sonnet-4-6": (3.0, 15.0),    # ($/MTok in, $/MTok out) — placeholder
        "claude-haiku-4-5-20251001": (0.8, 4.0),
    }
    rate_in, rate_out = rates.get(model_id, (3.0, 15.0))
    cost = (resp.usage.input_tokens * rate_in + resp.usage.output_tokens * rate_out) / 1_000_000

    return ModelOutput(extractions=extractions, raw_response=resp, cost_estimate_usd=cost)


def claude_sonnet_extract(client: Anthropic, transcript: str, **kw) -> ModelOutput:
    return _claude_extract(client, "claude-sonnet-4-6", transcript, **kw)


def claude_haiku_extract(client: Anthropic, transcript: str, **kw) -> ModelOutput:
    return _claude_extract(client, "claude-haiku-4-5-20251001", transcript, **kw)


# --- OpenAI ---

def gpt4o_extract(client: OpenAI, transcript: str, temperature: float = 0.3) -> ModelOutput:
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        tools=[OPENAI_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_extractions"}},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(transcript=transcript)},
        ],
    )
    msg = resp.choices[0].message
    extractions: list[dict] = []
    if msg.tool_calls:
        import json as _json
        try:
            extractions = _json.loads(msg.tool_calls[0].function.arguments).get("extractions", [])
        except Exception:
            extractions = []

    # Placeholder rates — update with current OpenAI pricing.
    rate_in, rate_out = 5.0, 15.0
    usage = resp.usage
    cost = (usage.prompt_tokens * rate_in + usage.completion_tokens * rate_out) / 1_000_000

    return ModelOutput(extractions=extractions, raw_response=resp, cost_estimate_usd=cost)


# --- Registry ---

MODELS = {
    "claude-sonnet-4-6": claude_sonnet_extract,
    "claude-haiku-4-5": claude_haiku_extract,
    "gpt-4o": gpt4o_extract,
}
