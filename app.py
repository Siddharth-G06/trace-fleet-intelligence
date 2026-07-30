"""
TRACE — Fleet Intelligence  |  Module 1: Fleet Health Overview

Streamlit entry-point for the TRACE hypothesis-driven fleet intelligence
system.  This first module provides a real-time fleet health dashboard
backed by the NASA CMAPSS FD001 dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Ensure project root is on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.data_layer.loader import compute_rul, load_raw_data
from src.data_layer.health_score import compute_health_scores, get_fleet_snapshot

# ── Page configuration (must be first Streamlit call) ─────────────────────────
cfg = load_config()
st_cfg = cfg["streamlit"]
st.set_page_config(
    page_title=st_cfg["page_title"],
    page_icon=st_cfg["page_icon"],
    layout=st_cfg["layout"],
    initial_sidebar_state="collapsed",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    }

    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(88, 166, 255, 0.3);
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.08);
    }
    [data-testid="stMetricValue"] { font-size: 2.4rem !important; font-weight: 700 !important; }

    h1 {
        background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    hr { border-color: rgba(255,255,255,0.1) !important; margin: 0.5rem 0 1.5rem !important; }
    h3 {
        color: #8b949e !important; font-weight: 500 !important;
        font-size: 0.85rem !important; text-transform: uppercase !important;
        letter-spacing: 1.5px !important; margin-top: 1.5rem !important;
    }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading fleet telemetry…")
def load_fleet_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process the training set into health-scored DataFrames."""
    rul_cap: int = cfg["data"]["rul_cap"]
    df_train, _, _ = load_raw_data()
    df_train = compute_rul(df_train)
    df_scored = compute_health_scores(df_train, rul_cap)
    snapshot = get_fleet_snapshot(df_scored)
    return df_scored, snapshot


# ── Sidebar: cache control ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Clears cached data and reloads the pipeline.")

df_scored, fleet_snapshot = load_fleet_data()

# ── KPI counts ─────────────────────────────────────────────────────────────────
tier_counts   = fleet_snapshot["risk_tier"].value_counts()
n_total       = len(fleet_snapshot)
n_healthy     = int(tier_counts.get("HEALTHY",  0))
n_watch       = int(tier_counts.get("WATCH",    0))
n_warning     = int(tier_counts.get("WARNING",  0))
n_critical    = int(tier_counts.get("CRITICAL", 0))
n_at_risk     = n_warning + n_critical


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("TRACE — Fleet Intelligence")
st.caption("Hypothesis-driven vehicle health monitoring  ·  NASA CMAPSS FD001")
st.divider()


# ── KPI Metrics ────────────────────────────────────────────────────────────────
st.markdown("### Fleet Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🚗  Total Vehicles", n_total,
              help="Total engine units in FD001 dataset")
with col2:
    st.metric("🟢  Healthy", n_healthy,
              delta=f"{n_healthy/n_total*100:.0f}% of fleet",
              delta_color="normal", help="Health score ≥ 80")
with col3:
    st.metric("🟡  Watch", n_watch,
              delta=f"{n_watch/n_total*100:.0f}% of fleet",
              delta_color="off", help="Health score 50–79")
with col4:
    st.metric("🔴  Critical / Warning", n_at_risk,
              delta=f"{n_at_risk/n_total*100:.0f}% of fleet",
              delta_color="inverse", help="Health score < 50")

st.divider()


# ── Fleet Health Table ─────────────────────────────────────────────────────────
st.markdown("### Vehicle Health Status")

display_df = fleet_snapshot.rename(columns={
    "unit_id":       "Vehicle ID",
    "health_score":  "Health Score",
    "risk_tier":     "Risk Tier",
    "current_cycle": "Cycles Run",
    "rul_remaining": "RUL Remaining",
})

_TIER_COLORS = {
    "HEALTHY":  "color: #3fb950; font-weight: 600;",
    "WATCH":    "color: #d29922; font-weight: 600;",
    "WARNING":  "color: #f0883e; font-weight: 600;",
    "CRITICAL": "color: #f85149; font-weight: 700;",
}
_SCORE_BG = {
    "HEALTHY":  "background-color: rgba(63,185,80,0.08);",
    "WATCH":    "background-color: rgba(210,153,34,0.10);",
    "WARNING":  "background-color: rgba(240,136,62,0.10);",
    "CRITICAL": "background-color: rgba(248,81,73,0.12);",
}

def _style_tier(val: str) -> str:
    return _TIER_COLORS.get(val, "")

def _style_row(row: pd.Series) -> list[str]:
    return [_SCORE_BG.get(row["Risk Tier"], "")] * len(row)

styled = (
    display_df.style
    .apply(_style_row, axis=1)
    .map(_style_tier, subset=["Risk Tier"])
    .format({"Health Score": "{:.1f}", "RUL Remaining": "{:.0f}"})
)
st.dataframe(styled, width="stretch", height=420, hide_index=True)

st.divider()


# ── Fleet Health Bar Chart (Plotly — color-coded by risk tier) ─────────────────
st.markdown("### Fleet Health Score — All Vehicles")
st.caption(
    "Each bar = one vehicle. Height = health score (0–100). "
    "Color = risk tier.  Sorted by vehicle ID."
)

_TIER_BAR_COLOR = {
    "HEALTHY":  "#3fb950",
    "WATCH":    "#d29922",
    "WARNING":  "#f0883e",
    "CRITICAL": "#f85149",
}

# Sort by vehicle ID for a consistent left→right layout
plot_df = fleet_snapshot.sort_values("unit_id").reset_index(drop=True)

fig = go.Figure()

# One trace per tier so the legend shows tier names with correct colors
for tier, color in _TIER_BAR_COLOR.items():
    mask = plot_df["risk_tier"] == tier
    if not mask.any():
        continue
    sub = plot_df[mask]
    fig.add_trace(go.Bar(
        x=sub["unit_id"],
        y=sub["health_score"],
        name=tier,
        marker_color=color,
        hovertemplate=(
            "<b>Vehicle %{x}</b><br>"
            "Health Score: %{y:.1f}<br>"
            "RUL Remaining: %{customdata[0]:.0f} cycles<br>"
            "Cycles Run: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata=sub[["rul_remaining", "current_cycle"]].values,
    ))

fig.update_layout(
    barmode="overlay",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c9d1d9"),
    xaxis=dict(
        title="Vehicle ID",
        showgrid=False,
        tickmode="linear",
        dtick=5,
        color="#8b949e",
    ),
    yaxis=dict(
        title="Health Score (0–100)",
        range=[0, 105],
        gridcolor="rgba(255,255,255,0.06)",
        color="#8b949e",
    ),
    legend=dict(
        title="Risk Tier",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12),
    ),
    # Reference lines for tier thresholds
    shapes=[
        dict(type="line", x0=0, x1=1, xref="paper", y0=80, y1=80,
             line=dict(color="#3fb950", dash="dash", width=1)),
        dict(type="line", x0=0, x1=1, xref="paper", y0=50, y1=50,
             line=dict(color="#d29922", dash="dash", width=1)),
        dict(type="line", x0=0, x1=1, xref="paper", y0=20, y1=20,
             line=dict(color="#f85149", dash="dash", width=1)),
    ],
    annotations=[
        dict(x=101, y=80, xref="x", yref="y", text="Healthy ≥ 80",
             showarrow=False, font=dict(color="#3fb950", size=10), xanchor="left"),
        dict(x=101, y=50, xref="x", yref="y", text="Watch ≥ 50",
             showarrow=False, font=dict(color="#d29922", size=10), xanchor="left"),
        dict(x=101, y=20, xref="x", yref="y", text="Warning ≥ 20",
             showarrow=False, font=dict(color="#f85149", size=10), xanchor="left"),
    ],
    margin=dict(l=10, r=80, t=40, b=40),
    height=350,
)

st.plotly_chart(fig, width="stretch")


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TRACE · Module 1 — Data Foundation  ·  "
    "Built on NASA CMAPSS FD001  ·  "
    "Next: ML RUL prediction · Gemini reasoning · ChromaDB evidence"
)
