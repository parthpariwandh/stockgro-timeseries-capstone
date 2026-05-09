"""Streamlit dashboard layout for capstone outputs."""

from __future__ import annotations

import streamlit as st


DASHBOARD_SECTIONS = [
    "Stock selector",
    "Forecast viewer",
    "Portfolio allocation",
    "Volatility analysis",
    "Trading signals",
    "Model comparison",
    "Interactive charts",
    "Downloadable outputs",
]


def run_dashboard() -> None:
    """Render dashboard sections and placeholders."""
    st.set_page_config(page_title="TSA Capstone 2026 Dashboard", layout="wide")
    st.title("TSA Capstone 2026 — Time Series Analysis & Virtual Portfolio Management")
    st.caption("Consulting & Analytics Club, IIT Guwahati × StockGro")

    ticker = st.selectbox(
        "Select NSE stock",
        [
            "HDFCBANK.NS",
            "TCS.NS",
            "SUNPHARMA.NS",
            "HINDUNILVR.NS",
            "MARUTI.NS",
            "RELIANCE.NS",
        ],
    )
    st.success(f"Loaded dashboard view for {ticker}")

    for section in DASHBOARD_SECTIONS:
        with st.expander(section, expanded=False):
            st.write(
                "Section is scaffolded and ready for notebook/module-integrated outputs."
            )
