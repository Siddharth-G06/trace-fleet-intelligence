"""
TRACE — Hypothesis-Driven Fleet Intelligence System

Streamlit dashboard integrating all four TRACE modules:
  Module 1 — Data Foundation (loader, health scores, fleet snapshot)
  Module 2 — LSTM RUL Predictor (fleet predictions, per-engine timelines)
  Module 3 — Hypothesis Reasoner (Gemini-powered natural-language analysis)
  Module 4 — Evidence Layer (ChromaDB event store, RAG retrieval)

Architecture: question-first intelligence — the fleet manager's hypothesis
drives targeted ML analysis + sensor-level evidence retrieval, returning a
structured verdict rather than surfacing reactive threshold alerts.
"""

from __future__ import annotations

import json
import logging
import os
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

# ── Page configuration (must be first Streamlit call) ─────────────────────────
cfg = load_config()
st_cfg = cfg["streamlit"]
st.set_page_config(
    page_title=st_cfg["page_title"],
    page_icon=st_cfg["page_icon"],
    layout=st_cfg["layout"],
    initial_sidebar_state="expanded",
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
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; }

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
    .verdict-card {
        background: rgba(88,166,255,0.06);
        border: 1px solid rgba(88,166,255,0.18);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .status-dot-green { color: #3fb950; font-size: 0.95rem; }
    .status-dot-orange { color: #f0883e; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

logger = logging.getLogger(__name__)

# ── Tier colour constants ──────────────────────────────────────────────────────
_TIER_COLORS = {
    "HEALTHY":  "color: #3fb950; font-weight: 600;",
    "WATCH":    "color: #d29922; font-weight: 600;",
    "WARNING":  "color: #f0883e; font-weight: 600;",
    "CRITICAL": "color: #f85149; font-weight: 700;",
}
_TIER_BG = {
    "HEALTHY":  "background-color: rgba(63,185,80,0.08);",
    "WATCH":    "background-color: rgba(210,153,34,0.10);",
    "WARNING":  "background-color: rgba(240,136,62,0.10);",
    "CRITICAL": "background-color: rgba(248,81,73,0.12);",
}
_TIER_HEX = {
    "HEALTHY":  "#3fb950",
    "WATCH":    "#d29922",
    "WARNING":  "#f0883e",
    "CRITICAL": "#f85149",
}

EXAMPLE_QUESTIONS = [
    "Which vehicles will need maintenance in the next 30 cycles?",
    "Show me the 5 most at-risk vehicles right now",
    "Are Group A vehicles degrading faster than Group B?",
    "Which vehicles show unusual health patterns?",
    "Is fleet health improving or declining over the last 50 cycles?",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_training_metadata() -> dict[str, Any] | None:
    """Load training metadata JSON if it exists."""
    meta_path = Path(cfg["model"]["metadata_path"])
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _style_tier(val: str) -> str:
    return _TIER_COLORS.get(val, "")


def _style_row_bg(row: pd.Series, tier_col: str = "Risk Tier") -> list[str]:
    return [_TIER_BG.get(row[tier_col], "")] * len(row)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_pipeline() -> dict[str, Any]:
    """Run Module 1 data pipeline (cached)."""
    from src.data_layer.loader import run_pipeline
    return run_pipeline()


@st.cache_data(show_spinner=False)
def _load_health_timeline() -> pd.DataFrame:
    """Build full per-cycle health-scored DataFrame from training data (cached)."""
    from src.data_layer.loader import compute_rul, load_raw_data
    from src.data_layer.health_score import compute_health_scores
    rul_cap: int = cfg["data"]["rul_cap"]
    df_train, _, _ = load_raw_data()
    df_train = compute_rul(df_train)
    return compute_health_scores(df_train, rul_cap)


@st.cache_data(show_spinner=False)
def _load_fleet_snapshot(health_timeline: pd.DataFrame) -> pd.DataFrame:
    """Build live fleet snapshot (one row per engine, randomised lifecycle point)."""
    from src.data_layer.health_score import get_fleet_snapshot
    return get_fleet_snapshot(health_timeline)


@st.cache_resource(show_spinner=False)
def _load_predictor(n_features: int = 17):
    """Load trained LSTM Predictor (cached resource)."""
    try:
        from src.ml_layer.lstm_predictor import Predictor
        return Predictor(cfg, n_features=n_features)
    except Exception as exc:
        return exc


@st.cache_data(show_spinner=False)
def _build_predicted_fleet() -> tuple[pd.DataFrame | None, str | None]:
    """Run fleet-level LSTM predictions on the test set (cached)."""
    meta = _load_training_metadata()
    n_features = (meta.get("n_features", 17) if meta else 17)
    predictor = _load_predictor(n_features)
    if isinstance(predictor, Exception):
        return None, str(predictor)

    try:
        from src.ml_layer.lstm_predictor import build_sequences_with_unit_ids
        from src.data_layer.loader import drop_constant_sensors, load_raw_data
        import joblib

        data_cfg = cfg["data"]
        window_size: int = data_cfg["window_size"]

        _, df_test, df_rul = load_raw_data()
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
        df_test["RUL"] = df_test.groupby("unit_id")["RUL"].transform(lambda s: s.bfill())
        for uid, grp_idx in df_test.groupby("unit_id").groups.items():
            grp = df_test.loc[grp_idx]
            if grp["RUL"].isna().any():
                last_rul = grp["RUL"].dropna().iloc[-1]
                last_cycle = grp["cycle"].max()
                df_test.loc[grp_idx, "RUL"] = last_rul + (last_cycle - grp["cycle"]).values
        df_test["RUL"] = df_test["RUL"].clip(upper=data_cfg["rul_cap"]).astype(np.float32)
        df_test_clean = drop_constant_sensors(df_test)

        X_te, y_te, unit_ids = build_sequences_with_unit_ids(df_test_clean, window_size)

        scaler = joblib.load(data_cfg["scaler_path"])
        n, ws, nf = X_te.shape
        X_te = scaler.transform(X_te.reshape(-1, nf)).reshape(n, ws, nf).astype("float32")

        last_cycles_arr = np.zeros(len(unit_ids), dtype=np.int64)
        for i, uid in enumerate(unit_ids):
            grp = df_test_clean[df_test_clean["unit_id"] == uid].sort_values("cycle")
            last_cycles_arr[i] = int(grp["cycle"].iloc[-1])

        fleet_df = predictor.predict_fleet(X_te, unit_ids, last_cycles_arr)

        # Attach actual RUL + actual health score
        actual_ruls: list[dict] = []
        for uid in np.sort(np.unique(unit_ids)):
            mask = unit_ids == uid
            last_idx = np.where(mask)[0][-1]
            actual_ruls.append({"unit_id": int(uid), "actual_rul": float(y_te[last_idx])})
        actual_df = pd.DataFrame(actual_ruls)
        fleet_df = fleet_df.merge(actual_df, on="unit_id", how="left")
        rul_cap = float(data_cfg["rul_cap"])
        fleet_df["actual_health_score"] = (fleet_df["actual_rul"] / rul_cap * 100.0).clip(0, 100)

        return fleet_df, None

    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def _load_chroma_client():
    """Initialise persistent ChromaDB client (cached resource)."""
    try:
        import chromadb  # type: ignore
        persist_dir: str = cfg["evidence"]["chromadb_persist_dir"]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=persist_dir)
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner=False)
def _load_event_logger(_chroma_client, _timeline_key: int):
    """Initialise EventLogger and run idempotent setup (cached resource)."""
    try:
        from src.evidence_layer.event_logger import EventLogger
        timeline = st.session_state.get("health_timeline")
        if timeline is None:
            return Exception("health_timeline not yet initialised")
        el = EventLogger(cfg, _chroma_client)
        el.setup(timeline)
        return el
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner=False)
def _load_rag_retriever(_chroma_client, _embedder):
    """Initialise RAGRetriever (cached resource)."""
    try:
        from src.evidence_layer.rag_retriever import RAGRetriever
        return RAGRetriever(cfg, _chroma_client, embedder=_embedder)
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner=False)
def _load_reasoner(_fleet_hash: int, _rag_available: bool):
    """Instantiate HypothesisReasoner (cached resource)."""
    try:
        from src.reasoning_layer.hypothesis_reasoner import HypothesisReasoner
        fleet_df = st.session_state.get("predicted_fleet")
        timeline = st.session_state.get("health_timeline")
        rag = st.session_state.get("rag_retriever")
        if fleet_df is None:
            # Fallback: use snapshot
            snap = st.session_state.get("fleet_snapshot")
            if snap is not None:
                fleet_df = snap.rename(columns={
                    "health_score": "predicted_health_score",
                    "rul_remaining": "predicted_rul",
                    "current_cycle": "last_cycle",
                })
        return HypothesisReasoner(
            cfg, fleet_df,
            health_timeline_df=timeline,
            rag_retriever=rag if _rag_available else None,
        )
    except Exception as exc:
        return exc


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _initialise_session_state() -> None:
    """One-time heavy initialisation, wrapped in a spinner."""
    needs_init = "pipeline_data" not in st.session_state

    if needs_init:
        with st.spinner("Loading TRACE…"):
            # Module 1 — Data pipeline
            if "pipeline_data" not in st.session_state:
                try:
                    st.session_state["pipeline_data"] = _load_pipeline()
                except Exception as exc:
                    st.session_state["pipeline_data"] = None
                    logger.error("Pipeline failed: %s", exc)

            # Module 1 — Full health timeline (all cycles, all engines)
            if "health_timeline" not in st.session_state:
                try:
                    st.session_state["health_timeline"] = _load_health_timeline()
                except Exception as exc:
                    st.session_state["health_timeline"] = pd.DataFrame()
                    logger.error("Health timeline failed: %s", exc)

            # Module 1 — Fleet snapshot
            if "fleet_snapshot" not in st.session_state:
                ht = st.session_state.get("health_timeline")
                if ht is not None and not ht.empty:
                    st.session_state["fleet_snapshot"] = _load_fleet_snapshot(ht)
                else:
                    st.session_state["fleet_snapshot"] = pd.DataFrame()

            # Module 2 — Predicted fleet
            if "predicted_fleet" not in st.session_state:
                fleet_df, err = _build_predicted_fleet()
                st.session_state["predicted_fleet"] = fleet_df
                st.session_state["pred_error"] = err

            # Module 4 — ChromaDB client
            if "chroma_client" not in st.session_state:
                client = _load_chroma_client()
                st.session_state["chroma_client"] = (
                    client if not isinstance(client, Exception) else None
                )

            # Module 4 — Event Logger (idempotent setup)
            if "event_logger" not in st.session_state:
                client = st.session_state.get("chroma_client")
                if client is not None:
                    ht = st.session_state.get("health_timeline")
                    if ht is not None and not ht.empty:
                        el = _load_event_logger(client, id(ht))
                        st.session_state["event_logger"] = (
                            el if not isinstance(el, Exception) else None
                        )
                    else:
                        st.session_state["event_logger"] = None
                else:
                    st.session_state["event_logger"] = None

            # Module 4 — RAG Retriever
            if "rag_retriever" not in st.session_state:
                el = st.session_state.get("event_logger")
                client = st.session_state.get("chroma_client")
                if el is not None and client is not None:
                    embedder = getattr(el, "_embedder", None)
                    rag = _load_rag_retriever(client, embedder)
                    st.session_state["rag_retriever"] = (
                        rag if not isinstance(rag, Exception) else None
                    )
                else:
                    st.session_state["rag_retriever"] = None

            # Module 3 — Hypothesis Reasoner
            if "reasoner" not in st.session_state:
                fleet_df = st.session_state.get("predicted_fleet")
                rag = st.session_state.get("rag_retriever")
                fleet_hash = id(fleet_df) if fleet_df is not None else 0
                reasoner = _load_reasoner(fleet_hash, rag is not None)
                st.session_state["reasoner"] = (
                    reasoner if not isinstance(reasoner, Exception) else None
                )

    # Persistent UI state (always init if missing)
    for key, default in [
        ("last_verdict", None),
        ("current_question", ""),
        ("run_analysis", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


_initialise_session_state()

# ── Convenience references ─────────────────────────────────────────────────────
_health_timeline: pd.DataFrame = st.session_state.get("health_timeline", pd.DataFrame())
_fleet_snapshot: pd.DataFrame  = st.session_state.get("fleet_snapshot",  pd.DataFrame())
_predicted_fleet: pd.DataFrame | None = st.session_state.get("predicted_fleet")
_pred_error: str | None = st.session_state.get("pred_error")
_chroma_client = st.session_state.get("chroma_client")
_event_logger = st.session_state.get("event_logger")
_rag_retriever = st.session_state.get("rag_retriever")
_reasoner = st.session_state.get("reasoner")
_evidence_available = _rag_retriever is not None


# ── KPI counts ──────────────────────────────────────────────────────────────────
def _tier_counts(df: pd.DataFrame, tier_col: str = "risk_tier") -> dict[str, int]:
    if df.empty:
        return {"HEALTHY": 0, "WATCH": 0, "WARNING": 0, "CRITICAL": 0}
    vc = df[tier_col].value_counts()
    return {t: int(vc.get(t, 0)) for t in ("HEALTHY", "WATCH", "WARNING", "CRITICAL")}


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 TRACE")
    st.caption("Hypothesis-driven fleet intelligence")
    st.divider()

    # System Status
    st.markdown("### System Status")
    meta = _load_training_metadata()
    n_total = len(_fleet_snapshot)
    st.metric("Fleet Size", f"{n_total} vehicles")

    if _event_logger is not None:
        try:
            stats = _event_logger.get_event_stats()
            st.metric("Total Events", stats.get("total_events", 0))
        except Exception:
            st.metric("Total Events", "—")
    else:
        st.metric("Total Events", "—")

    if meta:
        rmse_val = meta.get("rmse")
        st.metric("Model RMSE", f"{rmse_val:.2f} cycles" if isinstance(rmse_val, (int, float)) else "—")
    else:
        st.metric("Model RMSE", "—")

    st.divider()

    # How TRACE Works
    st.markdown("### How TRACE Works")
    st.markdown("""
**Question → Analysis → Evidence → Verdict**

Most fleet systems surface alerts reactively.
TRACE inverts this — bring your hypothesis,
TRACE finds the evidence in your vehicle data.

1. Ask a question in natural language
2. TRACE parses it into a structured query
3. LSTM health analysis runs on relevant vehicles
4. Sensor events retrieved as supporting evidence
5. Structured verdict returned with recommended action
""")

    st.divider()

    # Gemini API Key status
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        st.markdown(
            "<span class='status-dot-green'>✅ Gemini API Key detected — AI synthesis active</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='status-dot-orange'>⚠️ GEMINI_API_KEY not set — rule-based fallback active</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.caption("Clears all cached state and reloads.")


# ── App header ──────────────────────────────────────────────────────────────────
st.title("TRACE — Fleet Intelligence")
st.caption("Hypothesis-driven vehicle health monitoring  ·  NASA CMAPSS FD001")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🛡️  Fleet Overview",
    "🧠  Fleet Assistant",
    "📊  Model Info",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Health Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Fleet Health Overview")

    # ── 4-metric row ──────────────────────────────────────────────────────────
    tc = _tier_counts(_fleet_snapshot)
    n_total = sum(tc.values()) or 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🚗 Total Vehicles", n_total)
    with c2:
        st.metric(
            "🟢 Healthy",
            tc["HEALTHY"],
            delta=f"{tc['HEALTHY']/n_total*100:.0f}% of fleet",
            delta_color="normal",
        )
    with c3:
        st.metric(
            "🟡 Watch",
            tc["WATCH"],
            delta=f"{tc['WATCH']/n_total*100:.0f}% of fleet",
            delta_color="off",
        )
    with c4:
        n_critical = tc["CRITICAL"] + tc["WARNING"]
        st.metric(
            "🔴 Critical",
            tc["CRITICAL"],
            delta=f"{tc['CRITICAL']/n_total*100:.0f}% of fleet",
            delta_color="inverse",
        )

    # ── Fleet health table ─────────────────────────────────────────────────────
    st.divider()

    # Build display dataframe
    if not _fleet_snapshot.empty:
        display_df = _fleet_snapshot.rename(columns={
            "unit_id":       "Vehicle ID",
            "health_score":  "Health Score",
            "risk_tier":     "Risk Tier",
            "current_cycle": "Cycles Run",
            "rul_remaining": "RUL Remaining",
        })

        # Merge predicted health score from Module 2 if available
        if _predicted_fleet is not None:
            pred_col = _predicted_fleet[["unit_id", "predicted_health_score", "predicted_rul"]].rename(
                columns={
                    "unit_id":                "Vehicle ID",
                    "predicted_health_score": "Predicted Health Score",
                    "predicted_rul":          "Predicted RUL",
                }
            )
            display_df = display_df.merge(pred_col, on="Vehicle ID", how="left")

        # Reorder columns for clarity
        base_cols = ["Vehicle ID", "Predicted Health Score", "Risk Tier", "Cycles Run", "Predicted RUL"]
        existing_cols = [c for c in base_cols if c in display_df.columns]
        extra_cols = [c for c in display_df.columns if c not in existing_cols]
        display_df = display_df[existing_cols + extra_cols]

        fmt: dict[str, str] = {
            "Health Score":           "{:.1f}",
            "RUL Remaining":          "{:.0f}",
        }
        if "Predicted Health Score" in display_df.columns:
            fmt["Predicted Health Score"] = "{:.1f}"
        if "Predicted RUL" in display_df.columns:
            fmt["Predicted RUL"] = "{:.1f}"

        styled = (
            display_df.style
            .apply(lambda row: _style_row_bg(row, "Risk Tier"), axis=1)
            .map(_style_tier, subset=["Risk Tier"])
            .format(fmt)
        )
        st.dataframe(styled, use_container_width=True, height=420, hide_index=True)
    else:
        st.info("Fleet data not available — check data pipeline.")

    # ── Health Score Distribution ──────────────────────────────────────────────
    st.subheader("Health Score Distribution")

    if not _fleet_snapshot.empty:
        score_col = "predicted_health_score" if (
            _predicted_fleet is not None and not _predicted_fleet.empty
        ) else "health_score"

        if _predicted_fleet is not None and not _predicted_fleet.empty:
            hist_df = _predicted_fleet[["unit_id", "predicted_health_score", "risk_tier"]].copy()
            hist_df.columns = ["unit_id", "health_score_val", "risk_tier"]
        else:
            hist_df = _fleet_snapshot[["unit_id", "health_score", "risk_tier"]].copy()
            hist_df.columns = ["unit_id", "health_score_val", "risk_tier"]

        fig_hist = go.Figure()
        bins = np.linspace(0, 100, 11)  # 10 buckets

        for tier, color in _TIER_HEX.items():
            sub = hist_df[hist_df["risk_tier"] == tier]["health_score_val"]
            if sub.empty:
                continue
            fig_hist.add_trace(go.Histogram(
                x=sub,
                name=tier,
                xbins=dict(start=0, end=100, size=10),
                marker_color=color,
                opacity=0.85,
                hovertemplate=f"<b>{tier}</b><br>Score range: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))

        fig_hist.update_layout(
            barmode="stack",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#c9d1d9"),
            xaxis=dict(
                title="Health Score (0–100)",
                gridcolor="rgba(255,255,255,0.06)",
                color="#8b949e",
                range=[0, 100],
            ),
            yaxis=dict(
                title="Vehicle Count",
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
            ),
            margin=dict(l=10, r=10, t=40, b=40),
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Vehicle Drill-Down ─────────────────────────────────────────────────────
    st.subheader("Vehicle Drill-Down")

    if not _fleet_snapshot.empty:
        # Sort options by health score ascending (worst first)
        sorted_units = (
            _fleet_snapshot.sort_values("health_score", ascending=True)["unit_id"]
            .tolist()
        )
        selected_unit = st.selectbox(
            "Select a vehicle to inspect",
            options=sorted_units,
            format_func=lambda x: f"Engine {x}",
            key="tab1_drill_unit",
        )

        if selected_unit:
            left_col, right_col = st.columns([2, 1])

            with left_col:
                # Health score timeline (ground truth across all cycles)
                gt = (
                    _health_timeline[_health_timeline["unit_id"] == selected_unit]
                    .sort_values("cycle")[["cycle", "health_score"]]
                    .reset_index(drop=True)
                )

                fig_drill = go.Figure()
                fig_drill.add_trace(go.Scatter(
                    x=gt["cycle"],
                    y=gt["health_score"],
                    mode="lines",
                    name="Ground Truth",
                    line=dict(color="#58a6ff", width=2),
                    hovertemplate="Cycle %{x}: Health = %{y:.1f}<extra></extra>",
                ))

                # Predicted point from Module 2
                if _predicted_fleet is not None:
                    pred_row = _predicted_fleet[_predicted_fleet["unit_id"] == selected_unit]
                    if not pred_row.empty:
                        pred_hs = float(pred_row["predicted_health_score"].iloc[0])
                        last_cyc = int(pred_row["last_cycle"].iloc[0])
                        fig_drill.add_trace(go.Scatter(
                            x=[last_cyc],
                            y=[pred_hs],
                            mode="markers",
                            name="LSTM Predicted",
                            marker=dict(color="#f78166", size=14, symbol="star",
                                        line=dict(width=1.5, color="#fff")),
                            hovertemplate=f"Cycle {last_cyc}: Predicted = {pred_hs:.1f}<extra></extra>",
                        ))

                fig_drill.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#c9d1d9"),
                    xaxis=dict(title="Cycle", gridcolor="rgba(255,255,255,0.06)", color="#8b949e"),
                    yaxis=dict(title="Health Score", range=[-5, 105],
                               gridcolor="rgba(255,255,255,0.06)", color="#8b949e"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    shapes=[
                        dict(type="line", x0=0, x1=1, xref="paper", y0=80, y1=80,
                             line=dict(color="#3fb950", dash="dot", width=1)),
                        dict(type="line", x0=0, x1=1, xref="paper", y0=50, y1=50,
                             line=dict(color="#d29922", dash="dot", width=1)),
                        dict(type="line", x0=0, x1=1, xref="paper", y0=20, y1=20,
                             line=dict(color="#f85149", dash="dot", width=1)),
                    ],
                    margin=dict(l=10, r=10, t=30, b=40),
                    height=300,
                    title=dict(text=f"Engine {selected_unit} — Health Timeline",
                               font=dict(size=12, color="#8b949e")),
                )
                st.plotly_chart(fig_drill, use_container_width=True)

            with right_col:
                snap_row = _fleet_snapshot[_fleet_snapshot["unit_id"] == selected_unit]
                if not snap_row.empty:
                    row = snap_row.iloc[0]
                    hs = float(row["health_score"])
                    tier = str(row["risk_tier"])
                    tier_color = {"HEALTHY": "#3fb950", "WATCH": "#d29922",
                                  "WARNING": "#f0883e", "CRITICAL": "#f85149"}.get(tier, "#8b949e")

                    # Health score gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=hs,
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor="#8b949e"),
                            bar=dict(color=tier_color),
                            bgcolor="rgba(255,255,255,0.04)",
                            steps=[
                                dict(range=[0, 20], color="rgba(248,81,73,0.15)"),
                                dict(range=[20, 50], color="rgba(240,136,62,0.10)"),
                                dict(range=[50, 80], color="rgba(210,153,34,0.10)"),
                                dict(range=[80, 100], color="rgba(63,185,80,0.10)"),
                            ],
                        ),
                        number=dict(suffix="/100", font=dict(size=22, color=tier_color)),
                    ))
                    fig_gauge.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#c9d1d9"),
                        margin=dict(l=10, r=10, t=20, b=10),
                        height=220,
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown(
                        f"<div style='text-align:center; font-size:1.1rem; font-weight:700; "
                        f"color:{tier_color}; margin-bottom:0.5rem;'>{tier}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Cycles Run", int(row["current_cycle"]))
                    st.metric("RUL Remaining", f"{row['rul_remaining']:.0f} cycles")

            # ── Event History ──────────────────────────────────────────────────
            st.subheader("Event History")
            if _rag_retriever is not None:
                with st.spinner(f"Fetching events for Engine {selected_unit}…"):
                    try:
                        vehicle_events = _rag_retriever.retrieve_for_vehicle(
                            unit_id=int(selected_unit), top_k=8
                        )
                    except Exception as ev_exc:
                        vehicle_events = []
                        st.warning(f"Event retrieval failed: {ev_exc}")

                if vehicle_events:
                    _EV_ICONS = {
                        "health_drop":       "⬇️",
                        "critical_entry":    "🔴",
                        "sustained_decline": "📉",
                        "recovery":          "🟢",
                    }
                    for ev in vehicle_events:
                        icon = _EV_ICONS.get(ev["event_type"], "📋")
                        ev_type_label = ev["event_type"].replace("_", " ").title()
                        expander_title = (
                            f"{icon} {ev_type_label} · "
                            f"Cycle {ev['cycle']} · "
                            f"Health: {ev['health_score_after']:.1f}"
                        )
                        with st.expander(expander_title, expanded=False):
                            st.markdown(ev["description"])
                            hc1, hc2 = st.columns(2)
                            hc1.metric("Health Score After", f"{ev['health_score_after']:.1f}")
                            hc2.metric("Risk Tier", ev["risk_tier"])
                else:
                    st.info(
                        f"No events found for Engine {selected_unit}. "
                        "Try selecting a different vehicle.",
                        icon="ℹ️",
                    )
            else:
                st.info(
                    "Evidence store not available — "
                    "run `python src/evidence_layer/event_logger.py` first.",
                    icon="ℹ️",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fleet Assistant
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Fleet Assistant")
    st.caption(
        "Ask a hypothesis about your fleet. TRACE will analyze the vehicle data "
        "and return a verdict with supporting evidence."
    )

    # ── Example question buttons ───────────────────────────────────────────────
    st.markdown("**Example questions — click to fill:**")
    eq_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, eq in zip(eq_cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(eq[:38] + "…", key=f"eq_{eq[:18]}", use_container_width=True):
                st.session_state["current_question"] = eq
                st.session_state["run_analysis"] = True
                st.rerun()

    # ── Text input + Analyze button ────────────────────────────────────────────
    question_input = st.text_input(
        "Your question",
        value=st.session_state.get("current_question", ""),
        placeholder="e.g. Which vehicles will need maintenance in the next 30 cycles?",
        key="assistant_question_input",
        label_visibility="collapsed",
    )

    analyze_col, _ = st.columns([1, 4])
    with analyze_col:
        analyze_clicked = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True,
            key="analyze_btn",
        )

    if analyze_clicked and question_input.strip():
        st.session_state["current_question"] = question_input.strip()
        st.session_state["run_analysis"] = True

    # ── Run analysis ────────────────────────────────────────────────────────────
    if st.session_state.get("run_analysis") and st.session_state.get("current_question", "").strip():
        question = st.session_state["current_question"].strip()
        st.session_state["run_analysis"] = False

        if _reasoner is None:
            st.error(
                "Fleet Assistant unavailable. Ensure all modules have been run "
                "and dependencies are installed.",
                icon="❌",
            )
        else:
            try:
                with st.spinner("Analysing your fleet…"):
                    verdict = _reasoner.analyze(question)
                st.session_state["last_verdict"] = verdict
            except Exception as exc:
                logger.error("analyze() failed: %s", exc, exc_info=True)
                st.error("Analysis failed. Please try again.", icon="❌")
                st.session_state["last_verdict"] = None

    # ── Display verdict ─────────────────────────────────────────────────────────
    verdict = st.session_state.get("last_verdict")

    if verdict is not None:
        # Unknown query type warning
        if verdict.query_type == "unknown":
            st.warning(
                "Could not parse this question. "
                "Try rephrasing or use one of the example questions above.",
                icon="⚠️",
            )
        else:
            st.divider()

            # Backend badge
            backend_label = "✨ Gemini" if verdict.used_gemini else "⚙️ Rule-based"
            st.markdown(
                f"<span style='font-size:0.78rem;color:#8b949e;'>"
                f"Query type: <code>{verdict.query_type}</code> &nbsp;·&nbsp; "
                f"Backend: {backend_label} &nbsp;·&nbsp; "
                f"Computed: {verdict.computed_at.strftime('%H:%M:%S')} UTC"
                f"</span>",
                unsafe_allow_html=True,
            )

            st.subheader("Verdict")
            st.info(verdict.verdict_summary, icon="🔍")

            # 3-metric row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Confirmed", "✅ Yes" if verdict.confirmed else "❌ No")
            with m2:
                st.metric("Confidence", f"{verdict.confidence:.0%}")
            with m3:
                st.metric("Vehicles Identified", len(verdict.supporting_vehicles))

            # Supporting vehicles table
            if verdict.supporting_vehicles:
                sv_rows = []
                ref_df = _predicted_fleet if _predicted_fleet is not None else None
                if ref_df is None and not _fleet_snapshot.empty:
                    ref_df = _fleet_snapshot.rename(columns={
                        "health_score":  "predicted_health_score",
                        "rul_remaining": "predicted_rul",
                    })
                for veh in verdict.supporting_vehicles:
                    try:
                        uid_num = int(str(veh).replace("Engine ", "").strip())
                        if ref_df is not None:
                            row = ref_df[ref_df["unit_id"] == uid_num]
                            if not row.empty:
                                sv_rows.append({
                                    "Vehicle":       veh,
                                    "Health Score":  round(float(row["predicted_health_score"].iloc[0]), 1),
                                    "Predicted RUL": round(float(row["predicted_rul"].iloc[0]), 1),
                                    "Risk Tier":     str(row["risk_tier"].iloc[0]),
                                })
                                continue
                    except Exception:
                        pass
                    sv_rows.append({
                        "Vehicle": veh, "Health Score": "N/A",
                        "Predicted RUL": "N/A", "Risk Tier": "N/A",
                    })
                st.dataframe(pd.DataFrame(sv_rows), hide_index=True, use_container_width=True)

            # Key metrics as a table
            if verdict.key_metrics:
                km_rows = []
                for k, v in verdict.key_metrics.items():
                    if not isinstance(v, (list, dict, pd.DataFrame)):
                        km_rows.append({"Metric": k.replace("_", " ").title(), "Value": str(v)})
                if km_rows:
                    st.table(pd.DataFrame(km_rows))

            st.success(f"**Recommended Action:** {verdict.recommended_action}", icon="⚡")

            # ── Evidence Trail ─────────────────────────────────────────────────
            st.divider()
            st.subheader("Evidence Trail")
            st.caption(
                "Specific sensor events retrieved from vehicle history "
                "that support this verdict"
            )

            _EV_ICONS2 = {
                "health_drop":       "⬇️",
                "critical_entry":    "🔴",
                "sustained_decline": "📉",
                "recovery":          "🟢",
            }

            if verdict.evidence_events:
                for rank, ev in enumerate(verdict.evidence_events, 1):
                    icon = _EV_ICONS2.get(ev.get("event_type", ""), "📋")
                    ev_type_label = ev.get("event_type", "event").replace("_", " ").title()
                    score = ev.get("relevance_score", 0.0)
                    expander_title = (
                        f"{rank}. {ev_type_label} · "
                        f"Engine {ev.get('unit_id')} · "
                        f"Cycle {ev.get('cycle')} · "
                        f"Relevance: {score:.2f}"
                    )
                    with st.expander(f"{icon} {expander_title}", expanded=False):
                        st.markdown(ev.get("description", ""))
                        ec1, ec2 = st.columns(2)
                        ec1.metric("Health Before/After",
                                   f"{ev.get('health_score_after', 0):.1f}")
                        ec2.metric("Risk Tier", ev.get("risk_tier", "N/A"))
            else:
                st.info(
                    "No specific events retrieved for this query. "
                    "Verdict is based on aggregate health analysis.",
                    icon="ℹ️",
                )

    elif not st.session_state.get("run_analysis"):
        if verdict is None:
            # Placeholder
            st.markdown(
                """
                <div style='
                    margin-top: 2.5rem;
                    padding: 3rem;
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
                        TRACE parses your hypothesis, runs targeted ML analysis,<br>
                        and returns a verdict grounded in real vehicle sensor data.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Info
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model & System Information")

    # Load training metadata at render time
    meta = _load_training_metadata()

    # ── LSTM Performance ───────────────────────────────────────────────────────
    st.subheader("LSTM Performance")

    if meta:
        pm1, pm2, pm3, pm4 = st.columns(4)
        rmse_v = meta.get("rmse")
        mae_v  = meta.get("mae")
        ep_v   = meta.get("epochs_trained")
        ws_v   = meta.get("window_size")

        with pm1:
            st.metric("Test RMSE", f"{rmse_v:.2f} cycles" if isinstance(rmse_v, (int, float)) else "—")
        with pm2:
            st.metric("Test MAE", f"{mae_v:.2f} cycles" if isinstance(mae_v, (int, float)) else "—")
        with pm3:
            st.metric("Epochs Trained", ep_v if ep_v is not None else "—")
        with pm4:
            st.metric("Window Size", f"{ws_v} cycles" if ws_v is not None else "—")

        # Predicted vs Actual RUL scatter
        if _predicted_fleet is not None and "actual_rul" in _predicted_fleet.columns:
            st.markdown("#### Predicted vs Actual RUL (Test Set)")
            rul_cap_val = float(cfg["data"]["rul_cap"])
            fig_scatter = go.Figure()

            # Diagonal reference line
            fig_scatter.add_trace(go.Scatter(
                x=[0, rul_cap_val],
                y=[0, rul_cap_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
                hoverinfo="skip",
            ))

            for tier, color in _TIER_HEX.items():
                mask = _predicted_fleet["risk_tier"] == tier
                if not mask.any():
                    continue
                sub = _predicted_fleet[mask]
                fig_scatter.add_trace(go.Scatter(
                    x=sub["actual_rul"],
                    y=sub["predicted_rul"],
                    mode="markers",
                    name=tier,
                    marker=dict(color=color, size=9, opacity=0.85,
                                line=dict(width=1, color="rgba(255,255,255,0.2)")),
                    hovertemplate=(
                        "<b>Engine %{customdata[0]}</b><br>"
                        "Actual RUL: %{x:.1f}<br>"
                        "Predicted RUL: %{y:.1f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=sub[["unit_id"]].values,
                ))

            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#c9d1d9"),
                xaxis=dict(title="Actual RUL (cycles)", gridcolor="rgba(255,255,255,0.06)",
                           color="#8b949e", range=[-5, rul_cap_val + 10]),
                yaxis=dict(title="Predicted RUL (cycles)", gridcolor="rgba(255,255,255,0.06)",
                           color="#8b949e", range=[-5, rul_cap_val + 10]),
                legend=dict(title="Risk Tier", orientation="h",
                            yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=40),
                height=420,
                title=dict(text="Predicted vs Actual RUL (Test Set)",
                           font=dict(size=13, color="#8b949e")),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info(
            "No trained model found. Run:\n\n"
            "`python src/ml_layer/lstm_predictor.py`",
            icon="ℹ️",
        )

    # ── Per-Vehicle Health Timeline ────────────────────────────────────────────
    st.subheader("Per-Vehicle Health Timeline")

    if not _health_timeline.empty:
        all_units = sorted(_health_timeline["unit_id"].unique().tolist())
        selected_engine = st.selectbox(
            "Choose engine",
            options=all_units,
            format_func=lambda x: f"Engine {x}",
            key="model_tab_engine_select",
        )

        if selected_engine:
            gt_tl = (
                _health_timeline[_health_timeline["unit_id"] == selected_engine]
                .sort_values("cycle")[["cycle", "health_score"]]
                .reset_index(drop=True)
            )

            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=gt_tl["cycle"],
                y=gt_tl["health_score"],
                mode="lines",
                name="Ground Truth",
                line=dict(color="#58a6ff", width=2.5),
                hovertemplate="Cycle %{x}: Health = %{y:.1f}<extra></extra>",
            ))

            if _predicted_fleet is not None:
                pred_row = _predicted_fleet[_predicted_fleet["unit_id"] == selected_engine]
                if not pred_row.empty:
                    p_hs  = float(pred_row["predicted_health_score"].iloc[0])
                    p_cyc = int(pred_row["last_cycle"].iloc[0])
                    fig_tl.add_trace(go.Scatter(
                        x=[p_cyc],
                        y=[p_hs],
                        mode="markers",
                        name="Predicted",
                        marker=dict(color="#f78166", size=14, symbol="star",
                                    line=dict(width=1.5, color="#fff")),
                        hovertemplate=f"Cycle {p_cyc}: Predicted = {p_hs:.1f}<extra></extra>",
                    ))

            fig_tl.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#c9d1d9"),
                xaxis=dict(title="Cycle", gridcolor="rgba(255,255,255,0.06)", color="#8b949e"),
                yaxis=dict(title="Health Score (0–100)", range=[-5, 105],
                           gridcolor="rgba(255,255,255,0.06)", color="#8b949e"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                shapes=[
                    dict(type="line", x0=0, x1=1, xref="paper", y0=80, y1=80,
                         line=dict(color="#3fb950", dash="dot", width=1)),
                    dict(type="line", x0=0, x1=1, xref="paper", y0=50, y1=50,
                         line=dict(color="#d29922", dash="dot", width=1)),
                    dict(type="line", x0=0, x1=1, xref="paper", y0=20, y1=20,
                         line=dict(color="#f85149", dash="dot", width=1)),
                ],
                margin=dict(l=10, r=10, t=40, b=40),
                height=360,
                title=dict(text=f"Engine {selected_engine} — Ground Truth vs Predicted",
                           font=dict(size=13, color="#8b949e")),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

    st.divider()

    # ── System Architecture ────────────────────────────────────────────────────
    st.subheader("System Architecture")
    st.markdown("""
Traditional fleet monitoring surfaces alerts reactively. TRACE inverts this —
the manager's question drives the analysis. The system parses the hypothesis,
runs targeted ML on the relevant vehicle subset, retrieves supporting sensor
events, and returns a structured verdict.

| Module | Component | Role |
|--------|-----------|------|
| **1** | Data Foundation | Loads NASA CMAPSS FD001, computes RUL, assigns health scores (0–100) and risk tiers |
| **2** | LSTM Predictor | Two-layer LSTM trained to predict remaining useful life; converts RUL → health score |
| **3** | Hypothesis Reasoner | Gemini parses natural-language question → structured query → pandas analysis → verdict |
| **4** | Evidence Layer | ChromaDB + sentence-transformers event store; RAG retrieval for sensor-level evidence |
| **5** | Dashboard | Streamlit UI — Fleet Overview, Fleet Assistant, Model Info |

**Data flow:**

```
Fleet Manager's Question
         ↓
 Hypothesis Reasoner  ← Gemini / rule-based parser
         ↓
 LSTM Analysis Engine  ← 5 query handlers on fleet DataFrames
         ↓
 RAG Evidence Retrieval  ← ChromaDB semantic search over event logs
         ↓
 Structured VerdictResult  (confirmed, confidence, supporting vehicles, action)
         ↓
 Streamlit Dashboard
```
""")

    # ── Event Store Statistics ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Event Store Statistics")

    if _event_logger is not None:
        try:
            ev_stats = _event_logger.get_event_stats()
            by_type = ev_stats.get("events_per_type", {})

            es1, es2, es3, es4 = st.columns(4)
            with es1:
                st.metric("Total Events", ev_stats.get("total_events", 0))
            with es2:
                st.metric("Health Drop Events", by_type.get("health_drop", 0))
            with es3:
                st.metric("Critical Entry Events", by_type.get("critical_entry", 0))
            with es4:
                st.metric("Sustained Decline Events", by_type.get("sustained_decline", 0))
        except Exception as exc:
            st.warning(f"Could not load event statistics: {exc}")
    else:
        st.info(
            "Evidence store not initialised. "
            "Run `python src/evidence_layer/event_logger.py` to populate.",
            icon="ℹ️",
        )


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TRACE · Modules 1–4 — Data Foundation · LSTM RUL Prediction · "
    "Hypothesis Reasoner · RAG Evidence Layer  ·  "
    "Built on NASA CMAPSS FD001"
)
