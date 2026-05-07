"""Streamlit dashboard for eval results.

Reads results/latest.json. Run `python run_evals.py` first to populate it."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_PATH = Path(__file__).parent / "results" / "latest.json"

st.set_page_config(page_title="Audit", page_icon="📊", layout="wide")
st.title("Audit")
st.caption("Eval comparison across models on the Project 2 extraction task.")

if not RESULTS_PATH.exists():
    st.warning("No results yet. Run `python run_evals.py` first.")
    st.stop()

raw = json.loads(RESULTS_PATH.read_text())
df = pd.DataFrame(raw["metrics"])
if df.empty:
    st.warning("Results file is empty.")
    st.stop()


# --- Aggregate by model ---

st.subheader("Model comparison (mean ± stddev across runs and cases)")

agg = df.groupby("model").agg(
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
    recall_mean=("recall", "mean"),
    precision_mean=("precision", "mean"),
    qa_mean=("quote_attribution", "mean"),
    cost_mean=("cost_usd", "mean"),
).reset_index()

agg["f1"] = agg.apply(lambda r: f"{r['f1_mean']:.2f} ± {r['f1_std']:.2f}", axis=1)
agg["recall"] = agg["recall_mean"].map("{:.2f}".format)
agg["precision"] = agg["precision_mean"].map("{:.2f}".format)
agg["quote_attribution"] = agg["qa_mean"].map("{:.2%}".format)
agg["cost_per_transcript"] = agg["cost_mean"].map("${:.4f}".format)

st.dataframe(
    agg[["model", "f1", "recall", "precision", "quote_attribution", "cost_per_transcript"]],
    hide_index=True,
    use_container_width=True,
)


# --- Per-case heatmap ---

st.subheader("F1 by case × model")

pivot = df.pivot_table(index="case_id", columns="model", values="f1", aggfunc="mean")
st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1), use_container_width=True)


# --- Variance check ---

st.subheader("Variance — does the model give consistent results?")
st.caption("Models with high stddev are less reliable in production, even if mean F1 is high.")

variance = df.groupby("model")["f1"].agg(["mean", "std", "min", "max"]).reset_index()
variance.columns = ["model", "mean F1", "stddev", "min F1", "max F1"]
st.dataframe(variance, hide_index=True, use_container_width=True)


# --- Drilldown ---

st.subheader("Drilldown")
col1, col2 = st.columns(2)
with col1:
    selected_model = st.selectbox("Model", sorted(df["model"].unique()))
with col2:
    selected_case = st.selectbox("Case", sorted(df["case_id"].unique()))

drill = df[(df["model"] == selected_model) & (df["case_id"] == selected_case)]
st.dataframe(drill, hide_index=True, use_container_width=True)

# Raw extractions for this cell
raw_for_cell = [
    r for r in raw["raw_outputs"]
    if r["model"] == selected_model and r["case_id"] == selected_case
]
if raw_for_cell:
    st.write("**Extractions from runs:**")
    for r in raw_for_cell:
        with st.expander(f"Run {r['run_idx']+1}"):
            for e in r["extracted"]:
                st.markdown(f"- **{e.get('category', '?')}** (sev {e.get('severity', '?')}): {e.get('claim', '')}")
                st.caption(f"  Quote: \"{e.get('verbatim_quote', '')}\"")

st.caption(f"Eval ran in {raw['elapsed_s']:.1f}s. Results: {RESULTS_PATH}")
