from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


def get_backend_base_url():
    return os.getenv("COREP_BACKEND_HOST", "http://localhost:8000").rstrip("/")


def call_backend(scenario: str):
    base_url = get_backend_base_url()
    url = f"{base_url}/api/analyze_scenario"
    try:
        response = requests.post(url, json={"scenario": scenario}, timeout=60)
    except Exception as exc:
        st.error(f"Error contacting backend at {url}: {exc}")
        return None

    if not response.ok:
        st.error(f"Backend returned HTTP {response.status_code}: {response.text}")
        return None

    try:
        return response.json()
    except Exception as exc:
        st.error(f"Failed to parse backend response as JSON: {exc}")
        return None


def render_corep_table(corep_result: Dict[str, Any]):
    cet1 = corep_result.get("CET1")
    at1 = corep_result.get("AT1")
    tier2 = corep_result.get("Tier2")
    rwa = corep_result.get("RWA")
    cet1_ratio = corep_result.get("CET1_ratio")

    rows = [
        {"Metric": "CET1 (Common Equity Tier 1)", "Amount": cet1},
        {"Metric": "AT1 (Additional Tier 1)", "Amount": at1},
        {"Metric": "Tier 2 capital", "Amount": tier2},
        {"Metric": "Total RWA", "Amount": rwa},
        {"Metric": "CET1 ratio (CET1 / RWA, decimal)", "Amount": cet1_ratio},
    ]

    st.subheader("COREP Own Funds summary")
    st.table(rows)


def render_validation_warnings(corep_result: Dict[str, Any]):
    warnings = corep_result.get("validation_warnings") or []
    missing_fields = corep_result.get("missing_fields") or []

    if not warnings and not missing_fields:
        st.success("No validation warnings. All key numeric fields appear present.")
        return

    if warnings:
        st.warning("Validation warnings:")
        for w in warnings:
            st.markdown(f"- {w}")

    if missing_fields:
        st.info("Missing or uncertain numeric fields identified:")
        st.markdown(", ".join(sorted(set(str(f) for f in missing_fields))))


def main():
    st.set_page_config(
        page_title="PRA COREP Own Funds LLM Assistant",
        layout="wide",
    )

    st.title("PRA COREP Own Funds – LLM-assisted Reporting Prototype")
    st.caption(
        "Prototype assistant using RAG over synthetic PRA / COREP text. "
        "Not for actual regulatory reporting or compliance."
    )

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            f"**Backend URL**: `{get_backend_base_url()}`",
            help="Configure via COREP_BACKEND_HOST environment variable.",
        )

    st.markdown("### 1. Describe the banking scenario")
    default_example = (
        "A UK bank has total risk-weighted assets (RWA) of 5,000 million. "
        "Its CET1 capital is 300 million, AT1 instruments amount to 50 million, "
        "and Tier 2 capital is 80 million. Assess the COREP Own Funds position."
    )

    scenario = st.text_area(
        "Natural-language scenario",
        value=default_example,
        height=200,
    )

    if st.button("Analyse scenario", type="primary"):
        if not scenario.strip():
            st.error("Please provide a scenario description.")
            return

        with st.spinner("Running RAG retrieval, LLM analysis, and validation..."):
            result = call_backend(scenario)

        if not result:
            return

        st.markdown("### 2. Input and retrieved regulatory context")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User scenario")
            st.write(result.get("scenario", ""))

        with col2:
            st.subheader("Retrieved regulatory text")
            retrieved_context = result.get("retrieved_context") or []
            if not retrieved_context:
                st.info("No regulatory context retrieved.")
            else:
                for idx, doc in enumerate(retrieved_context):
                    title = f"[{idx + 1}] {doc.get('citation', 'citation unknown')}"
                    with st.expander(title, expanded=(idx == 0)):
                        st.markdown(f"**Source**: {doc.get('source', 'unknown')}")
                        st.write(doc.get("text", ""))

        st.markdown("### 3. Structured COREP Own Funds JSON")
        corep_result = result.get("corep_result") or {}
        st.code(json.dumps(corep_result, indent=2), language="json")

        st.markdown("### 4. COREP-style Own Funds table and checks")
        render_corep_table(corep_result)
        render_validation_warnings(corep_result)

        st.markdown("### 5. Audit trail and rules used")
        rules_used = corep_result.get("rules_used") or []
        if rules_used:
            st.markdown("**Rules / paragraphs cited:**")
            for r in rules_used:
                st.markdown(f"- {r}")
        else:
            st.info("No explicit regulatory rules were cited in this assessment.")

        raw_model_output = result.get("raw_model_output")
        with st.expander("Raw LLM JSON (pre-validation)", expanded=False):
            st.code(json.dumps(raw_model_output, indent=2), language="json")


if __name__ == "__main__":
    main()
