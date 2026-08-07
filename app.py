"""
TRACE — Fleet Intelligence  |  Modules 1–3: Fleet Health + LSTM Predictions + AI Reasoning

Streamlit entry-point for the TRACE hypothesis-driven fleet intelligence
system.  Module 1 provides the real-time fleet health dashboard.  Module 2
adds LSTM RUL predictions and per-engine health timelines.  Module 3
integrates the Gemini-powered HypothesisReasoner for natural-language
fleet intelligence queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        padding: 0.6rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading (Module 1) ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading fleet telemetry…")
def load_fleet_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process the training set into health-scored DataFrames."""
    rul_cap: int = cfg["data"]["rul_cap"]
    df_train, _, _ = load_raw_data()
    df_train = compute_rul(df_train)
    df_scored = compute_health_scores(df_train, rul_cap)
    snapshot = get_fleet_snapshot(df_scored)
    return df_scored, snapshot


# ── Predictor (Module 2) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LSTM predictor…")
def load_predictor():
    """Load the trained LSTM predictor (cached as a resource)."""
    try:
        from src.ml_layer.lstm_predictor import Predictor
        
        # Determine correct feature size from metadata to prevent mismatch
        meta = _load_training_metadata()
        n_features = meta.get("n_features", 17) if meta else 17
        
        predictor = Predictor(cfg, n_features=n_features)
        return predictor
    except Exception as exc:
        return exc


@st.cache_data(show_spinner="Running LSTM fleet predictions…")
def load_fleet_predictions() -> tuple[pd.DataFrame | None, str | None]:
    """Build test-set fleet predictions, returning (fleet_df, error_msg)."""
    predictor = load_predictor()
    if isinstance(predictor, Exception):
        return None, str(predictor)

    try:
        from src.ml_layer.lstm_predictor import build_sequences_with_unit_ids
        from src.data_layer.loader import (
            compute_rul, drop_constant_sensors, load_raw_data,
        )
        import joblib

        data_cfg = cfg["data"]
        window_size: int = data_cfg["window_size"]

        df_train, df_test, df_rul = load_raw_data()

        # Rebuild test with RUL column (same logic as loader.run_pipeline)
        df_test = df_test.copy()
        df_rul_idx = df_rul.reset_index(drop=True)
        df_rul_idx["unit_id"] = df_rul_idx.index + 1

        last_cycles_df = df_test.groupby("unit_id")["cycle"].max().reset_index()
        last_cycles_df = last_cycles_df.merge(
            df_rul_idx.rename(columns={"rul": "RUL"}), on="unit_id"
        )
        df_test = df_test.merge(
            last_cycles_df[["unit_id", "cycle", "RUL"]],
            on=["unit_id", "cycle"],
            how="left",
        )
        df_test["RUL"] = df_test.groupby("unit_id")["RUL"].transform(
            lambda s: s.bfill()
        )
        for uid, grp_idx in df_test.groupby("unit_id").groups.items():
            grp = df_test.loc[grp_idx]
            if grp["RUL"].isna().any():
                last_rul   = grp["RUL"].dropna().iloc[-1]
                last_cycle = grp["cycle"].max()
                df_test.loc[grp_idx, "RUL"] = (
                    last_rul + (last_cycle - grp["cycle"]).values
                )
        df_test["RUL"] = (
            df_test["RUL"].clip(upper=data_cfg["rul_cap"]).astype(np.float32)
        )
        df_test_clean = drop_constant_sensors(df_test)

        X_te, y_te, unit_ids = build_sequences_with_unit_ids(df_test_clean, window_size)

        # Normalize with persisted scaler
        scaler = joblib.load(data_cfg["scaler_path"])
        n, ws, nf = X_te.shape
        X_te = scaler.transform(X_te.reshape(-1, nf)).reshape(n, ws, nf).astype("float32")

        # last_cycles per window
        last_cycles_arr = np.zeros(len(unit_ids), dtype=np.int64)
        for i, uid in enumerate(unit_ids):
            grp = df_test_clean[df_test_clean["unit_id"] == uid].sort_values("cycle")
            last_cycles_arr[i] = int(grp["cycle"].iloc[-1])

        fleet_df = predictor.predict_fleet(X_te, unit_ids, last_cycles_arr)

        # Attach actual RUL (last window per unit)
        actual_ruls: list[dict] = []
        for uid in np.sort(np.unique(unit_ids)):
            mask     = unit_ids == uid
            last_idx = np.where(mask)[0][-1]
            actual_ruls.append({"unit_id": int(uid), "actual_rul": float(y_te[last_idx])})
        actual_df = pd.DataFrame(actual_ruls)
        fleet_df  = fleet_df.merge(actual_df, on="unit_id", how="left")

        # Ground-truth health score (same linear mapping as Module 1)
        rul_cap = float(data_cfg["rul_cap"])
        fleet_df["actual_health_score"] = (
            fleet_df["actual_rul"] / rul_cap * 100.0
        ).clip(0.0, 100.0)

        return fleet_df, None

    except Exception as exc:
        return None, str(exc)


def _load_training_metadata() -> dict[str, Any] | None:
    """Load training metadata JSON if it exists."""
    meta_path = Path(cfg["model"]["metadata_path"])
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Clears cached data and reloads the pipeline.")

    if st.button("🧠  Reset Assistant", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.caption("Re-initialises the Fleet Assistant with any new API key.")

    # ── Module 2: Training Metadata ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 LSTM Model Metrics")
    meta = _load_training_metadata()
    if meta:
        st.metric("RMSE", f"{meta.get('rmse', 'N/A'):.2f}" if isinstance(meta.get("rmse"), float) else "N/A")
        st.metric("MAE",  f"{meta.get('mae',  'N/A'):.2f}" if isinstance(meta.get("mae"),  float) else "N/A")
        st.metric("Epochs Trained", meta.get("epochs_trained", "N/A"))
        st.caption(f"**Model:** `{meta.get('model_path', 'N/A')}`")
        if meta.get("trained_at"):
            st.caption(f"**Trained at:** {meta['trained_at'][:19].replace('T', ' ')} UTC")
    else:
        st.info(
            "No trained model found. Run:\n\n"
            "`python -m src.ml_layer.lstm_predictor`",
            icon="ℹ️",
        )

df_scored, fleet_snapshot = load_fleet_data()
fleet_pred_df, pred_error = load_fleet_predictions()

# ── KPI counts ─────────────────────────────────────────────────────────────────
tier_counts = fleet_snapshot["risk_tier"].value_counts()
n_total     = len(fleet_snapshot)
n_healthy   = int(tier_counts.get("HEALTHY",  0))
n_watch     = int(tier_counts.get("WATCH",    0))
n_warning   = int(tier_counts.get("WARNING",  0))
n_critical  = int(tier_counts.get("CRITICAL", 0))
n_at_risk   = n_warning + n_critical


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


# ── Module 3: Reasoner loader ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising Fleet Assistant…")
def load_reasoner(_fleet_df, _timeline_df, _rag_retriever):
    """Instantiate HypothesisReasoner (cached as a resource)."""
    try:
        from src.reasoning_layer.hypothesis_reasoner import HypothesisReasoner
        return HypothesisReasoner(
            cfg, _fleet_df,
            health_timeline_df=_timeline_df,
            rag_retriever=_rag_retriever,
        )
    except Exception as exc:
        return exc


# ── Module 4: Evidence Layer ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def load_chroma_client():
    """Initialise a persistent ChromaDB client (cached as a resource)."""
    try:
        import chromadb  # type: ignore
        from pathlib import Path
        persist_dir: str = cfg["evidence"]["chromadb_persist_dir"]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=persist_dir)
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner="Setting up Evidence Logger…")
def load_event_logger(_chroma_client, _timeline_df):
    """Initialise EventLogger and run idempotent setup (cached as a resource)."""
    try:
        from src.evidence_layer.event_logger import EventLogger
        el = EventLogger(cfg, _chroma_client)
        el.setup(_timeline_df)
        return el
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner="Loading RAG Retriever…")
def load_rag_retriever(_chroma_client, _embedder):
    """Initialise RAGRetriever (cached as a resource)."""
    try:
        from src.evidence_layer.rag_retriever import RAGRetriever
        return RAGRetriever(cfg, _chroma_client, embedder=_embedder)
    except Exception as exc:
        return exc


# ─── Initialise evidence layer ────────────────────────────────────────────────
_chroma_client = load_chroma_client()
_event_logger = None
_rag_retriever = None
_evidence_available = False
if not isinstance(_chroma_client, Exception):
    _event_logger = load_event_logger(_chroma_client, df_scored)
    if not isinstance(_event_logger, Exception):
        _embedder = getattr(_event_logger, "_embedder", None)
        _rag_retriever = load_rag_retriever(_chroma_client, _embedder)
        if not isinstance(_rag_retriever, Exception):
            _evidence_available = True

# ═══════════════════════════════════════════════════════════════════════════════
# TABS — Module 1 | Module 2 | Module 3 | Module 4
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🛡️  Fleet Health Overview",
    "🤖  Predicted vs Actual",
    "🧠  Fleet Assistant",
])


# ── TAB 1: Fleet Health Overview (Module 1 content) ───────────────────────────
with tab1:
    # ── Fleet Health Table ─────────────────────────────────────────────────────
    st.markdown("### Vehicle Health Status")

    display_df = fleet_snapshot.rename(columns={
        "unit_id":       "Vehicle ID",
        "health_score":  "Health Score",
        "risk_tier":     "Risk Tier",
        "current_cycle": "Cycles Run",
        "rul_remaining": "RUL Remaining",
    })

    # Merge in predicted health score if available
    if fleet_pred_df is not None:
        pred_col = fleet_pred_df[["unit_id", "predicted_health_score"]].rename(
            columns={
                "unit_id":                "Vehicle ID",
                "predicted_health_score": "Predicted Health Score",
            }
        )
        display_df = display_df.merge(pred_col, on="Vehicle ID", how="left")

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

    fmt_dict: dict[str, str] = {
        "Health Score":   "{:.1f}",
        "RUL Remaining":  "{:.0f}",
    }
    if "Predicted Health Score" in display_df.columns:
        fmt_dict["Predicted Health Score"] = "{:.1f}"

    styled = (
        display_df.style
        .apply(_style_row, axis=1)
        .map(_style_tier, subset=["Risk Tier"])
        .format(fmt_dict)
    )
    st.dataframe(styled, width="stretch", height=420, hide_index=True)

    st.divider()

    # ── Fleet Health Bar Chart ─────────────────────────────────────────────────
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

    plot_df = fleet_snapshot.sort_values("unit_id").reset_index(drop=True)
    fig = go.Figure()

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

    # ── Per-vehicle drill-down (Module 4) ─────────────────────────────────────
    if _evidence_available and _rag_retriever is not None:
        st.divider()
        st.markdown("### 🔎 Vehicle Event History")
        st.caption(
            "Select a vehicle to inspect its full health-event history "
            "retrieved from the evidence store."
        )
        all_unit_ids_t1 = sorted(fleet_snapshot["unit_id"].unique().tolist())
        drill_unit = st.selectbox(
            "Inspect a specific vehicle",
            options=all_unit_ids_t1,
            format_func=lambda x: f"Engine {x}",
            key="tab1_drill_selectbox",
        )
        if drill_unit:
            with st.spinner(f"Fetching events for Engine {drill_unit}…"):
                vehicle_events = _rag_retriever.retrieve_for_vehicle(
                    unit_id=int(drill_unit), top_k=8
                )
            if vehicle_events:
                for ev in vehicle_events:
                    icon = {
                        "health_drop":      "⬇️",
                        "critical_entry":   "🔴",
                        "sustained_decline": "📉",
                        "recovery":         "🟢",
                    }.get(ev["event_type"], "📋")
                    label = (
                        f"{icon} {ev['event_type'].replace('_', ' ').title()} · "
                        f"Cycle {ev['cycle']} · "
                        f"Relevance: {ev['relevance_score']:.2f}"
                    )
                    with st.expander(label, expanded=False):
                        st.write(ev["description"])
                        ec1, ec2 = st.columns(2)
                        ec1.metric("Health Score", f"{ev['health_score_after']:.1f}")
                        ec2.metric("Risk Tier", ev["risk_tier"])
            else:
                st.info(
                    f"No events found for Engine {drill_unit} — "
                    "try a different vehicle or refresh the evidence store."
                )

# ── TAB 2: Predicted vs Actual (Module 2 content) ─────────────────────────────
with tab2:
    if pred_error:
        st.error(
            f"⚠️  LSTM model not available: {pred_error}\n\n"
            "Train the model first by running:\n"
            "```\npython -m src.ml_layer.lstm_predictor\n```",
        )
        st.stop()

    if fleet_pred_df is None:
        st.warning("No predictions available. Please train the LSTM model first.")
        st.stop()

    # ── RMSE / MAE metrics ─────────────────────────────────────────────────────
    st.markdown("### Model Accuracy Metrics")
    if meta:
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("📉  RMSE", f"{meta['rmse']:.2f}", help="Root Mean Squared Error on test set")
        with mc2:
            st.metric("📊  MAE",  f"{meta['mae']:.2f}",  help="Mean Absolute Error on test set")
        with mc3:
            st.metric("⏱️  Epochs Trained", meta.get("epochs_trained", "—"))
    st.divider()

    # ── Predicted vs Actual Scatter ────────────────────────────────────────────
    st.markdown("### Predicted vs Actual RUL")
    st.caption(
        "Each point = one test engine unit at its last observed cycle. "
        "The diagonal line shows perfect prediction."
    )

    _TIER_SCATTER = {
        "HEALTHY":  "#3fb950",
        "WATCH":    "#d29922",
        "WARNING":  "#f0883e",
        "CRITICAL": "#f85149",
    }

    fig_scatter = go.Figure()

    # ── Diagonal reference line ────────────────────────────────────────────────
    rul_cap_val = float(cfg["data"]["rul_cap"])
    fig_scatter.add_trace(go.Scatter(
        x=[0, rul_cap_val],
        y=[0, rul_cap_val],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    # ── Points colored by risk tier ────────────────────────────────────────────
    for tier, color in _TIER_SCATTER.items():
        mask = fleet_pred_df["risk_tier"] == tier
        if not mask.any():
            continue
        sub = fleet_pred_df[mask]
        fig_scatter.add_trace(go.Scatter(
            x=sub["actual_rul"],
            y=sub["predicted_rul"],
            mode="markers",
            name=tier,
            marker=dict(
                color=color,
                size=10,
                opacity=0.85,
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
            ),
            hovertemplate=(
                "<b>Engine %{customdata[0]}</b><br>"
                "Actual RUL:    %{x:.1f}<br>"
                "Predicted RUL: %{y:.1f}<br>"
                "Health Score:  %{customdata[1]:.1f}<br>"
                "<extra></extra>"
            ),
            customdata=sub[["unit_id", "predicted_health_score"]].values,
        ))

    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(
            title="Actual RUL (cycles)",
            gridcolor="rgba(255,255,255,0.06)",
            color="#8b949e",
            range=[-5, rul_cap_val + 10],
        ),
        yaxis=dict(
            title="Predicted RUL (cycles)",
            gridcolor="rgba(255,255,255,0.06)",
            color="#8b949e",
            range=[-5, rul_cap_val + 10],
        ),
        legend=dict(
            title="Risk Tier",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=10, t=40, b=40),
        height=420,
    )
    st.plotly_chart(fig_scatter, width="stretch")

    st.divider()

    # ── Fleet health table with both scores ────────────────────────────────────
    st.markdown("### Fleet Health — Ground Truth vs Predicted")
    combined = fleet_pred_df[
        ["unit_id", "actual_rul", "actual_health_score",
         "predicted_rul", "predicted_health_score", "risk_tier", "last_cycle"]
    ].copy()
    combined = combined.sort_values("predicted_health_score").reset_index(drop=True)
    combined.columns = [
        "Vehicle ID", "Actual RUL", "GT Health Score",
        "Predicted RUL", "Predicted Health Score", "Risk Tier", "Last Cycle",
    ]

    def _tier_color(val: str) -> str:
        return _TIER_COLORS.get(val, "")

    def _combined_row_bg(row: pd.Series) -> list[str]:
        return [_SCORE_BG.get(row["Risk Tier"], "")] * len(row)

    styled_combined = (
        combined.style
        .apply(_combined_row_bg, axis=1)
        .map(_tier_color, subset=["Risk Tier"])
        .format({
            "Actual RUL":            "{:.1f}",
            "GT Health Score":       "{:.1f}",
            "Predicted RUL":         "{:.1f}",
            "Predicted Health Score":"{:.1f}",
        })
    )
    st.dataframe(styled_combined, width="stretch", height=380, hide_index=True)

    st.divider()

    # ── Per-engine health score timeline ───────────────────────────────────────
    st.markdown("### Health Score Timeline — Individual Engine")
    st.caption(
        "Select an engine to compare ground-truth vs predicted health score "
        "across all of its training cycles."
    )

    all_unit_ids = sorted(df_scored["unit_id"].unique().tolist())
    selected_unit = st.selectbox(
        "Choose Engine Unit",
        options=all_unit_ids,
        index=0,
        format_func=lambda x: f"Engine {x}",
        key="engine_selectbox",
    )

    # Ground-truth timeline from df_scored
    gt_timeline = (
        df_scored[df_scored["unit_id"] == selected_unit]
        .sort_values("cycle")[["cycle", "health_score", "RUL"]]
        .copy()
    )

    fig_timeline = go.Figure()

    # Ground truth line
    fig_timeline.add_trace(go.Scatter(
        x=gt_timeline["cycle"],
        y=gt_timeline["health_score"],
        mode="lines",
        name="Ground Truth",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="Cycle %{x}: GT Health = %{y:.1f}<extra></extra>",
    ))

    # Predicted health score: only 1 point per engine in the current setup
    # (predict_fleet uses only the last window). If available, show as a marker.
    if fleet_pred_df is not None:
        pred_row = fleet_pred_df[fleet_pred_df["unit_id"] == selected_unit]
        if not pred_row.empty:
            pred_hs  = float(pred_row["predicted_health_score"].iloc[0])
            last_cyc = int(pred_row["last_cycle"].iloc[0])
            fig_timeline.add_trace(go.Scatter(
                x=[last_cyc],
                y=[pred_hs],
                mode="markers",
                name="LSTM Predicted",
                marker=dict(
                    color="#f78166",
                    size=14,
                    symbol="star",
                    line=dict(width=1.5, color="#fff"),
                ),
                hovertemplate=(
                    f"Cycle {last_cyc}: Predicted Health = {pred_hs:.1f}"
                    "<extra></extra>"
                ),
            ))

    # Tier threshold reference lines
    fig_timeline.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(
            title="Cycle",
            gridcolor="rgba(255,255,255,0.06)",
            color="#8b949e",
        ),
        yaxis=dict(
            title="Health Score (0–100)",
            range=[-5, 105],
            gridcolor="rgba(255,255,255,0.06)",
            color="#8b949e",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=[
            dict(type="line", x0=0, x1=1, xref="paper", y0=80, y1=80,
                 line=dict(color="#3fb950", dash="dot", width=1)),
            dict(type="line", x0=0, x1=1, xref="paper", y0=50, y1=50,
                 line=dict(color="#d29922", dash="dot", width=1)),
            dict(type="line", x0=0, x1=1, xref="paper", y0=20, y1=20,
                 line=dict(color="#f85149", dash="dot", width=1)),
        ],
        annotations=[
            dict(x=gt_timeline["cycle"].max(), y=80, xref="x", yref="y",
                 text="Healthy", showarrow=False,
                 font=dict(color="#3fb950", size=9), xanchor="right"),
            dict(x=gt_timeline["cycle"].max(), y=50, xref="x", yref="y",
                 text="Watch",   showarrow=False,
                 font=dict(color="#d29922", size=9), xanchor="right"),
            dict(x=gt_timeline["cycle"].max(), y=20, xref="x", yref="y",
                 text="Warning", showarrow=False,
                 font=dict(color="#f85149", size=9), xanchor="right"),
        ],
        margin=dict(l=10, r=10, t=40, b=40),
        height=360,
        title=dict(
            text=f"Engine {selected_unit} — Health Score Over Time",
            font=dict(size=13, color="#8b949e"),
        ),
    )
    st.plotly_chart(fig_timeline, width="stretch")


# ── TAB 3: Fleet Assistant (Module 3 content) ─────────────────────────────────
with tab3:
    # ── Additional CSS for Fleet Assistant tab ────────────────────────────────
    st.markdown(
        """
        <style>
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .verdict-box {
            background: rgba(88,166,255,0.06);
            border: 1px solid rgba(88,166,255,0.2);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Fleet Assistant")
    st.caption(
        "Ask any question about your fleet in plain English. "
        "TRACE will analyse real vehicle data and return a grounded verdict."
    )

    # ── Example question chips ────────────────────────────────────────────────
    EXAMPLE_QUESTIONS = [
        "Which vehicles will need maintenance in the next 30 cycles?",
        "Show me the 5 most at-risk vehicles right now",
        "Are Group A vehicles degrading faster than Group B?",
        "Which vehicles show unusual health patterns?",
        "Is fleet health improving or declining over the last 50 cycles?",
    ]

    if "fleet_question" not in st.session_state:
        st.session_state["fleet_question"] = ""

    st.markdown("**Example questions — click to fill:**")
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, eq in zip(chip_cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(eq[:40] + "…", key=f"chip_{eq[:20]}", use_container_width=True):
                st.session_state["fleet_question"] = eq

    st.divider()

    # ── Question input ────────────────────────────────────────────────────────
    question_input = st.text_input(
        "Ask a question about your fleet…",
        value=st.session_state["fleet_question"],
        placeholder="e.g. Which vehicles will need maintenance in the next 30 cycles?",
        key="fleet_question_input",
        label_visibility="collapsed",
    )
    submit_col, _ = st.columns([1, 4])
    with submit_col:
        submitted = st.button(
            "🔍  Analyse",
            type="primary",
            use_container_width=True,
            key="fleet_submit",
        )

    if submitted and question_input.strip():
        # ── Load reasoner (uses df_scored as timeline, fleet_pred_df as fleet) ─
        timeline_df = df_scored
        reasoner_fleet_df = fleet_pred_df if fleet_pred_df is not None else None

        if reasoner_fleet_df is None:
            # Fallback: build a usable fleet_data from Module 1 snapshot
            snap = fleet_snapshot.rename(columns={
                "health_score":  "predicted_health_score",
                "rul_remaining": "predicted_rul",
                "current_cycle": "last_cycle",
            })
            reasoner_fleet_df = snap

        reasoner = load_reasoner(
            reasoner_fleet_df, timeline_df,
            _rag_retriever if _evidence_available else None,
        )

        if isinstance(reasoner, Exception):
            st.error(
                f"Fleet Assistant unavailable: {reasoner}\n\n"
                "Ensure `src/reasoning_layer/hypothesis_reasoner.py` is present."
            )
        else:
            with st.spinner("Analysing your fleet…"):
                verdict = reasoner.analyze(question_input.strip())

            # ── Verdict display ───────────────────────────────────────────────
            st.divider()

            # Backend badge
            backend_label = "✨ Gemini" if verdict.used_gemini else "⚙️ Rule-based"
            st.markdown(
                f"<span style='font-size:0.78rem;color:#8b949e;'>"
                f"Query type: <code>{verdict.query_type}</code> &nbsp;·&nbsp; "
                f"Backend: {backend_label} &nbsp;·&nbsp; "
                f"Computed at: {verdict.computed_at.strftime('%H:%M:%S')} UTC"
                f"</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Verdict summary (large info box)
            st.info(verdict.verdict_summary, icon="🔍")

            # Metric row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Confirmed",
                    "✅ Yes" if verdict.confirmed else "❌ No",
                    help="Whether the data supports the hypothesis in the question",
                )
            with m2:
                st.metric(
                    "Confidence",
                    f"{verdict.confidence * 100:.0f}%",
                    help="Parser confidence in query classification",
                )
            with m3:
                st.metric(
                    "Supporting Vehicles",
                    len(verdict.supporting_vehicles),
                    help="Number of vehicles directly cited in the analysis",
                )

            st.divider()

            # Supporting vehicles table
            if verdict.supporting_vehicles:
                st.markdown("### Supporting Vehicles")
                sv_rows = []
                for veh in verdict.supporting_vehicles:
                    # Try to look up health score from fleet_pred_df
                    try:
                        uid_num = int(veh.replace("Engine ", ""))
                        row = reasoner_fleet_df[
                            reasoner_fleet_df["unit_id"] == uid_num
                        ]
                        if not row.empty:
                            sv_rows.append({
                                "Vehicle": veh,
                                "Health Score": round(float(row["predicted_health_score"].iloc[0]), 1),
                                "Predicted RUL": round(float(row["predicted_rul"].iloc[0]), 1),
                                "Risk Tier": str(row["risk_tier"].iloc[0]),
                            })
                        else:
                            sv_rows.append({"Vehicle": veh, "Health Score": "N/A", "Predicted RUL": "N/A", "Risk Tier": "N/A"})
                    except Exception:
                        sv_rows.append({"Vehicle": veh, "Health Score": "N/A", "Predicted RUL": "N/A", "Risk Tier": "N/A"})

                sv_df = pd.DataFrame(sv_rows)
                st.dataframe(sv_df, hide_index=True, use_container_width=True)

            # Key metrics
            if verdict.key_metrics:
                st.markdown("### Key Metrics")
                st.json(verdict.key_metrics)

            # Recommended action (success box)
            st.success(f"**Recommended Action:** {verdict.recommended_action}", icon="⚡")

            # ── Evidence Trail (Module 4) ─────────────────────────────────────
            st.divider()
            st.subheader("🗂️ Evidence Trail")
            if verdict.evidence_events:
                st.caption(
                    f"{len(verdict.evidence_events)} relevant event(s) retrieved "
                    "from the fleet evidence store."
                )
                for ev in verdict.evidence_events:
                    icon = {
                        "health_drop":       "⬇️",
                        "critical_entry":    "🔴",
                        "sustained_decline": "📉",
                        "recovery":          "🟢",
                    }.get(ev["event_type"], "📋")
                    expander_title = (
                        f"{icon} {ev['event_type'].replace('_', ' ').title()} · "
                        f"Engine {ev['unit_id']} · "
                        f"Cycle {ev['cycle']} · "
                        f"Relevance: {ev['relevance_score']:.2f}"
                    )
                    with st.expander(expander_title, expanded=False):
                        st.write(ev["description"])
                        tc1, tc2 = st.columns(2)
                        tc1.metric("Health Score After", f"{ev['health_score_after']:.1f}")
                        tc2.metric("Risk Tier", ev["risk_tier"])
            else:
                if not _evidence_available:
                    st.info(
                        "Evidence store not available — ChromaDB or "
                        "sentence-transformers may not be installed.",
                        icon="ℹ️",
                    )
                else:
                    st.info(
                        "No specific events found for this query — "
                        "try a more specific question.",
                        icon="ℹ️",
                    )

    elif submitted and not question_input.strip():
        st.warning("Please enter a question before submitting.", icon="⚠️")

    if not submitted:
        # ── Placeholder when no question has been asked yet ───────────────────
        st.markdown(
            """
            <div style='
                margin-top: 2rem;
                padding: 2.5rem;
                border: 1px dashed rgba(255,255,255,0.12);
                border-radius: 14px;
                text-align: center;
                color: #484f58;
            '>
                <div style='font-size: 2.5rem; margin-bottom: 0.8rem;'>🧠</div>
                <div style='font-size: 1.1rem; font-weight: 600; color: #8b949e;
                    margin-bottom: 0.4rem;'>TRACE Fleet Assistant</div>
                <div style='font-size: 0.88rem;'>
                    Ask any natural-language question about your fleet.<br>
                    Powered by Gemini AI + real vehicle health data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TRACE · Modules 1–4 — Data Foundation · LSTM RUL Prediction · "
    "Gemini Fleet Assistant · RAG Evidence Layer  ·  "
    "Built on NASA CMAPSS FD001"
)
