"""
TRACE — Module 3: Hypothesis Reasoner

Converts natural-language fleet manager questions into structured
:class:`VerdictResult` objects grounded in real computed data.

Pipeline
--------
User question
  → ``_parse_question``  (Gemini temp=0, or rule-based fallback)
  → ``_run_analysis``    (pandas operations on fleet DataFrames)
  → ``_synthesize_verdict`` (Gemini temp=0.3, or template fallback)
  → :class:`VerdictResult`

All prompts and config values are read from ``config/config.yaml``.
All logging routes through :func:`~src.utils.logger.get_logger`.
No ``print`` statements are used inside classes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_PARSER_SYSTEM_PROMPT = """\
You are a query parser for TRACE, a fleet intelligence system.
Your job is to classify a fleet manager's question into a structured query.

Supported query types:
- fleet_comparison: comparing groups of vehicles (depots, regions)
- threshold_query: vehicles needing maintenance within N cycles
- anomaly_hunt: vehicles with unusual or irregular health patterns
- ranking_query: ranking vehicles by health, risk, or performance
- trend_analysis: whether fleet health is improving or declining over time

Respond ONLY with a valid JSON object (no markdown, no explanation):
{
  "query_type": "<one of the supported types or 'unknown'>",
  "params": {
    "threshold_cycles": <int or null>,
    "top_n": <int or null>,
    "time_window_cycles": <int or null>,
    "group_a_label": "<string or null>",
    "group_b_label": "<string or null>"
  },
  "confidence": <float between 0.0 and 1.0>
}
"""

_PARSER_USER_TEMPLATE = "Fleet manager question: \"{question}\""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are TRACE, an expert fleet intelligence assistant.
You have been given a fleet manager's question and the raw computed results of an analysis.
Write a concise, professional verdict based ONLY on the provided data — do not invent numbers.

Respond ONLY with a valid JSON object (no markdown, no explanation):
{
  "verdict_summary": "<2-3 sentences summarising what the data shows>",
  "recommended_action": "<one specific, actionable maintenance or operational recommendation>"
}
"""

_SYNTHESIS_USER_TEMPLATE = """\
Question: "{question}"
Query type: {query_type}
Computed results:
{results_json}
"""


# ── VerdictResult dataclass ───────────────────────────────────────────────────

@dataclass
class VerdictResult:
    """Structured verdict returned by :class:`HypothesisReasoner`.

    Attributes:
        original_question: The raw question submitted by the fleet manager.
        query_type: Classified query type (e.g. ``"ranking_query"``).
        confirmed: Whether the hypothesis embedded in the question is supported
            by the computed data.
        confidence: Parser confidence in the query classification (0.0–1.0).
        verdict_summary: Human-readable summary of the analysis findings.
        supporting_vehicles: List of vehicle IDs most relevant to the verdict.
        key_metrics: Dict of computed numeric metrics from the analysis.
        recommended_action: A specific, actionable recommendation.
        computed_at: UTC timestamp when the verdict was produced.
        used_gemini: Whether Gemini was used (True) or fallbacks fired (False).
    """

    original_question: str
    query_type: str
    confirmed: bool
    confidence: float
    verdict_summary: str
    supporting_vehicles: list[str]
    key_metrics: dict[str, Any]
    recommended_action: str
    computed_at: datetime
    used_gemini: bool = field(default=False)


# ── HypothesisReasoner ────────────────────────────────────────────────────────

class HypothesisReasoner:
    """Gemini-powered, hypothesis-driven fleet intelligence engine.

    Converts natural-language questions into :class:`VerdictResult` objects
    by chaining a Gemini parser → pandas analysis → Gemini synthesis, with
    a full rule-based / template fallback at every step.

    Args:
        config: Full parsed configuration dict (top-level from config.yaml).
        fleet_data: Output of ``Predictor.predict_fleet()`` — one row per
            unique ``unit_id`` with columns:
            ``unit_id``, ``predicted_rul``, ``predicted_health_score``,
            ``risk_tier``, ``last_cycle``.
        health_timeline_df: Optional. Full per-cycle health score DataFrame
            from Module 1 ``compute_health_scores()`` — many rows per engine
            with columns ``unit_id``, ``cycle``, ``health_score``.
            Required for trend analysis and anomaly detection.
            Falls back gracefully to fleet_data if not provided.

    Example::

        reasoner = HypothesisReasoner(cfg, fleet_df, timeline_df)
        result = reasoner.analyze("Which vehicles need maintenance soon?")
        print(result.verdict_summary)
    """

    def __init__(
        self,
        config: dict[str, Any],
        fleet_data: pd.DataFrame,
        health_timeline_df: pd.DataFrame | None = None,
    ) -> None:
        self._cfg = config
        self._gcfg = config.get("gemini", {})
        self._rcfg = config.get("reasoning", {})
        self._fleet_data = fleet_data.copy()
        self._timeline_df = (
            health_timeline_df.copy() if health_timeline_df is not None else None
        )

        # ── Gemini client ─────────────────────────────────────────────────────
        self._gemini_model: Any = None
        self._gemini_available = False
        self._init_gemini()

        logger.info(
            "HypothesisReasoner initialised  fleet_units=%d  timeline=%s  "
            "gemini_available=%s",
            len(self._fleet_data),
            "yes" if self._timeline_df is not None else "no",
            self._gemini_available,
        )

    # ── Gemini initialisation ─────────────────────────────────────────────────

    def _init_gemini(self) -> None:
        """Initialise the Gemini generative model client.

        Reads the API key from ``os.environ['GEMINI_API_KEY']`` first,
        falling back to the ``gemini.api_key`` config value.  Sets
        ``_gemini_available = False`` and logs a warning instead of raising
        if the key is missing or the import fails.
        """
        try:
            import google.generativeai as genai  # type: ignore[import]

            api_key = os.environ.get("GEMINI_API_KEY", "") or self._gcfg.get(
                "api_key", ""
            )
            if not api_key:
                logger.warning(
                    "GEMINI_API_KEY not set — Gemini disabled; "
                    "rule-based + template fallbacks will be used."
                )
                return

            genai.configure(api_key=api_key)
            model_name: str = self._gcfg.get("model", "gemini-1.5-flash")
            self._gemini_model = genai.GenerativeModel(model_name)
            self._gemini_available = True
            logger.info("Gemini client ready  model=%s", model_name)

        except ImportError:
            logger.warning(
                "google-generativeai not installed — Gemini disabled."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini init failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def _maybe_reinit_gemini(self) -> None:
        """Re-attempt Gemini initialisation if it failed at construction time.

        This handles the common Streamlit pattern where ``st.cache_resource``
        creates the reasoner before the ``GEMINI_API_KEY`` env var is set.
        Calling this at the top of :meth:`analyze` ensures the key is picked
        up on the next request without requiring a server restart.
        """
        if not self._gemini_available:
            api_key = os.environ.get("GEMINI_API_KEY", "") or self._gcfg.get("api_key", "")
            if api_key:
                logger.info("Detected GEMINI_API_KEY after init — re-initialising Gemini")
                self._init_gemini()

    def analyze(self, question: str) -> VerdictResult:
        """Parse a question, run analysis, and synthesize a verdict.

        Args:
            question: Natural-language question from the fleet manager.

        Returns:
            A fully populated :class:`VerdictResult`.  On any unexpected
            failure the returned result has ``confirmed=False`` and
            ``verdict_summary`` describing the error.
        """
        self._maybe_reinit_gemini()  # pick up env var set after cache init
        logger.info("analyze() called  question=%r", question[:120])
        try:
            parsed = self._parse_question(question)
            results = self._run_analysis(parsed)
            verdict_summary, recommended_action = self._synthesize_verdict(
                question, parsed["query_type"], results
            )

            # Determine confirmed from results
            confirmed = self._infer_confirmed(parsed["query_type"], results)

            # Build supporting vehicle list
            supporting = self._extract_supporting_vehicles(
                parsed["query_type"], results
            )

            # Strip non-serialisable values from key_metrics
            key_metrics = {
                k: v
                for k, v in results.items()
                if not isinstance(v, (list, pd.DataFrame))
            }

            return VerdictResult(
                original_question=question,
                query_type=parsed["query_type"],
                confirmed=confirmed,
                confidence=float(parsed.get("confidence", 0.5)),
                verdict_summary=verdict_summary,
                supporting_vehicles=supporting,
                key_metrics=key_metrics,
                recommended_action=recommended_action,
                computed_at=datetime.now(timezone.utc),
                used_gemini=self._gemini_available,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("analyze() failed: %s", exc, exc_info=True)
            return VerdictResult(
                original_question=question,
                query_type="unknown",
                confirmed=False,
                confidence=0.0,
                verdict_summary=f"Analysis could not be completed: {exc}",
                supporting_vehicles=[],
                key_metrics={},
                recommended_action="Review system logs and retry.",
                computed_at=datetime.now(timezone.utc),
                used_gemini=False,
            )

    # ── Step 1: Parse ─────────────────────────────────────────────────────────

    def _parse_question(self, question: str) -> dict[str, Any]:
        """Classify the question into a structured query dict via Gemini.

        Falls back to :meth:`_rule_based_parser` if Gemini is unavailable
        or returns malformed JSON.

        Args:
            question: Raw fleet manager question.

        Returns:
            Dict with keys ``query_type``, ``params``, ``confidence``.
        """
        if self._gemini_available:
            try:
                import google.generativeai as genai  # type: ignore[import]

                parser_temp: float = float(
                    self._gcfg.get("parser_temperature", 0.0)
                )
                max_tokens: int = int(
                    self._gcfg.get("max_output_tokens", 500)
                )
                generation_config = genai.GenerationConfig(
                    temperature=parser_temp,
                    max_output_tokens=max_tokens,
                )
                user_msg = _PARSER_USER_TEMPLATE.format(question=question)
                response = self._gemini_model.generate_content(
                    [_PARSER_SYSTEM_PROMPT, user_msg],
                    generation_config=generation_config,
                )
                raw_text: str = response.text.strip()
                parsed = self._safe_json_parse(raw_text)
                if parsed and "query_type" in parsed:
                    logger.info(
                        "Gemini parser  query_type=%s  confidence=%.2f",
                        parsed.get("query_type"),
                        parsed.get("confidence", 0.0),
                    )
                    return self._normalise_parsed(parsed)
                logger.warning(
                    "Gemini parser returned unexpected JSON — falling back. "
                    "raw=%r",
                    raw_text[:200],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini parser call failed: %s — falling back", exc)

        result = self._rule_based_parser(question)
        logger.info(
            "Rule-based parser  query_type=%s  confidence=%.2f",
            result["query_type"],
            result["confidence"],
        )
        return result

    def _rule_based_parser(self, question: str) -> dict[str, Any]:
        """Keyword-matching fallback parser using word-stem matching.

        Uses substring stems rather than exact words so that common
        misspellings (e.g. "maintainence") and synonyms (e.g. "days",
        "repair", "fix") are handled robustly.

        Args:
            question: Raw fleet manager question.

        Returns:
            Dict with the same structure as the Gemini parser output.
        """
        q = question.lower()
        # Normalise common typos / contractions before matching
        q = q.replace("maintainence", "maintenance")  # very common misspelling
        q = q.replace("maintenence",  "maintenance")
        q = q.replace("mantainance",  "maintenance")
        q = q.replace("mantenance",   "maintenance")

        # Extract numeric hints (match first number in the question)
        numbers = re.findall(r"\b(\d+)\b", q)
        first_num = int(numbers[0]) if numbers else None

        params: dict[str, Any] = {
            "threshold_cycles": None,
            "top_n": None,
            "time_window_cycles": None,
            "group_a_label": None,
            "group_b_label": None,
        }

        # ── Threshold / maintenance query ─────────────────────────────────────
        # Stems: "maintain" covers maintenance/maintaining; "servic" covers
        # service/servicing; "schedul" covers schedule/scheduled; etc.
        _threshold_stems = (
            "maintain",  # maintenance, maintaining, maintainence
            "servic",    # service, servicing
            "repair",
            "overhaul",
            "schedul",   # schedule, scheduled
            "due",
            "soon",
            "upcoming",
            "next ",     # "next 30 cycles/days"
            "within",
            "days",      # "in the next 30 days"
            "cycle",     # cycles / cycle
            "fix",
            "inspect",   # inspection, inspect
            "pm",        # preventive maintenance abbreviation
        )
        # ── Ranking query ─────────────────────────────────────────────────────
        _ranking_stems = (
            "worst",
            "at-risk",
            "at risk",
            "critical",
            "most at",
            "top ",
            "bottom ",
            "highest risk",
            "lowest health",
            "most danger",
            "riskiest",
            "poorest",
            "bad",
        )
        # ── Trend analysis ────────────────────────────────────────────────────
        _trend_stems = (
            "trend",
            "improv",    # improving, improved
            "declin",    # declining, declined
            "over time",
            "over the last",
            "getting better",
            "getting worse",
            "worsening",
            "deterior",  # deteriorating
            "progress",
            "overall health",
        )
        # ── Anomaly hunt ──────────────────────────────────────────────────────
        _anomaly_stems = (
            "unusual",
            "anomal",    # anomaly, anomalies, anomalous
            "strange",
            "irregular",
            "outlier",
            "erratic",
            "weird",
            "unexpected",
            "odd",
            "spike",
        )
        # ── Fleet comparison ──────────────────────────────────────────────────
        _comparison_stems = (
            "group",
            "depot",
            " vs ",
            "versus",
            "compar",    # compare, comparison
            "between",
            "faster",
            "slower",
            "differ",    # difference, differently
            "group a",
            "group b",
        )

        if any(stem in q for stem in _threshold_stems):
            query_type = "threshold_query"
            # If the user said "days", treat it as cycles (1 cycle ≈ 1 day in CMAPSS)
            params["threshold_cycles"] = first_num or self._rcfg.get(
                "default_threshold_cycles", 30
            )
            confidence = 0.85

        elif any(stem in q for stem in _ranking_stems):
            query_type = "ranking_query"
            params["top_n"] = first_num or self._rcfg.get("default_top_n", 5)
            confidence = 0.80

        elif any(stem in q for stem in _trend_stems):
            query_type = "trend_analysis"
            params["time_window_cycles"] = first_num or 50
            confidence = 0.80

        elif any(stem in q for stem in _anomaly_stems):
            query_type = "anomaly_hunt"
            confidence = 0.75

        elif any(stem in q for stem in _comparison_stems):
            query_type = "fleet_comparison"
            params["group_a_label"] = "Group A"
            params["group_b_label"] = "Group B"
            confidence = 0.78

        else:
            query_type = "unknown"
            confidence = 0.30

        return {
            "query_type": query_type,
            "params": params,
            "confidence": confidence,
        }

    # ── Step 2: Dispatch analysis ─────────────────────────────────────────────

    def _run_analysis(self, parsed_query: dict[str, Any]) -> dict[str, Any]:
        """Dispatch to the correct analysis method.

        Args:
            parsed_query: Output of :meth:`_parse_question`.

        Returns:
            Raw results dict.  Content varies by query type.
        """
        query_type: str = parsed_query.get("query_type", "unknown")
        params: dict[str, Any] = parsed_query.get("params", {})

        dispatch = {
            "fleet_comparison":  self._analyze_fleet_comparison,
            "threshold_query":   self._analyze_threshold_query,
            "anomaly_hunt":      self._analyze_anomaly_hunt,
            "ranking_query":     self._analyze_ranking_query,
            "trend_analysis":    self._analyze_trend_analysis,
        }

        handler = dispatch.get(query_type)
        if handler is None:
            logger.warning("No analysis handler for query_type=%r", query_type)
            return {"note": f"Unsupported query type: {query_type}"}

        logger.info("Running analysis  query_type=%s", query_type)
        return handler(params)

    # ── Analysis methods ──────────────────────────────────────────────────────

    def _analyze_fleet_comparison(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compare average health decline rate between Group A and Group B.

        Group A = unit_ids 1–50, Group B = unit_ids 51–100 (from config).
        Decline rate per engine is computed as:
        ``(health_score at cycle N-window) - (health_score at cycle N) / window``
        then averaged per group.

        Args:
            params: Parser params dict (currently unused for this query type).

        Returns:
            Dict with group decline rates, faster group, difference, and
            per-group critical vehicle counts.
        """
        rcfg = self._rcfg
        ga_range: list[int] = rcfg.get("group_a_units", [1, 50])
        gb_range: list[int] = rcfg.get("group_b_units", [51, 100])
        window: int = int(rcfg.get("health_decline_window", 20))

        def _decline_rate(unit_ids: list[int]) -> float:
            """Compute mean decline rate across a set of engines."""
            rates: list[float] = []
            if self._timeline_df is not None:
                for uid in unit_ids:
                    eng = (
                        self._timeline_df[self._timeline_df["unit_id"] == uid]
                        .sort_values("cycle")
                    )
                    if len(eng) < window + 1:
                        continue
                    hs_start = float(eng["health_score"].iloc[-window - 1])
                    hs_end = float(eng["health_score"].iloc[-1])
                    rates.append((hs_start - hs_end) / window)
            else:
                # Fallback: use predicted_health_score as proxy for single-point
                sub = self._fleet_data[
                    self._fleet_data["unit_id"].isin(unit_ids)
                ]
                # Estimate decline proportional to (100 - health_score)
                rates = list(
                    (100.0 - sub["predicted_health_score"]) / 100.0
                )
            return float(np.mean(rates)) if rates else 0.0

        fleet = self._fleet_data
        ga_ids = fleet[
            fleet["unit_id"].between(ga_range[0], ga_range[1])
        ]["unit_id"].tolist()
        gb_ids = fleet[
            fleet["unit_id"].between(gb_range[0], gb_range[1])
        ]["unit_id"].tolist()

        ga_rate = _decline_rate(ga_ids)
        gb_rate = _decline_rate(gb_ids)
        faster = "Group A" if ga_rate > gb_rate else "Group B"
        difference = abs(ga_rate - gb_rate)

        ga_critical = int(
            fleet[
                fleet["unit_id"].isin(ga_ids) & (fleet["risk_tier"] == "CRITICAL")
            ].shape[0]
        )
        gb_critical = int(
            fleet[
                fleet["unit_id"].isin(gb_ids) & (fleet["risk_tier"] == "CRITICAL")
            ].shape[0]
        )

        logger.info(
            "Fleet comparison  ga_rate=%.4f  gb_rate=%.4f  faster=%s",
            ga_rate, gb_rate, faster,
        )
        return {
            "group_a_decline_rate": round(ga_rate, 4),
            "group_b_decline_rate": round(gb_rate, 4),
            "faster_group": faster,
            "difference": round(difference, 4),
            "group_a_vehicles_critical": ga_critical,
            "group_b_vehicles_critical": gb_critical,
            "group_a_size": len(ga_ids),
            "group_b_size": len(gb_ids),
            "window_cycles": window,
        }

    def _analyze_threshold_query(self, params: dict[str, Any]) -> dict[str, Any]:
        """Find vehicles with predicted RUL below a threshold.

        Args:
            params: Must contain ``threshold_cycles`` (int); falls back to
                ``reasoning.default_threshold_cycles`` from config.

        Returns:
            Dict with vehicle list, count, average RUL, and most critical.
        """
        threshold: int = int(
            params.get("threshold_cycles")
            or self._rcfg.get("default_threshold_cycles", 30)
        )
        df = self._fleet_data
        due = (
            df[df["predicted_rul"] < threshold]
            .sort_values("predicted_rul", ascending=True)
            .reset_index(drop=True)
        )
        vehicles: list[str] = [f"Engine {int(uid)}" for uid in due["unit_id"]]
        avg_rul: float = float(due["predicted_rul"].mean()) if not due.empty else 0.0
        most_critical: str = vehicles[0] if vehicles else "None"

        logger.info(
            "Threshold query  threshold=%d  count=%d  most_critical=%s",
            threshold, len(vehicles), most_critical,
        )
        return {
            "vehicles_needing_maintenance": vehicles,
            "count": len(vehicles),
            "avg_rul": round(avg_rul, 1),
            "most_critical_vehicle": most_critical,
            "threshold_cycles": threshold,
        }

    def _analyze_anomaly_hunt(self, params: dict[str, Any]) -> dict[str, Any]:
        """Detect engines with high health-score variance across their timeline.

        An engine is anomalous if its variance exceeds:
        ``fleet_mean_variance + 1.5 * fleet_std_variance``

        Falls back to using last-cycle predicted health score dispersion if
        no timeline DataFrame is available.

        Args:
            params: Unused for this query type.

        Returns:
            Dict with anomalous vehicle IDs, their variances, and fleet mean.
        """
        if self._timeline_df is not None and not self._timeline_df.empty:
            variances: dict[int, float] = (
                self._timeline_df.groupby("unit_id")["health_score"]
                .var()
                .to_dict()
            )
        else:
            # Fallback: synthesise single-point variance proxy
            df = self._fleet_data
            # Use predicted_health_score — variance is estimated per engine
            # as zero (single point); flag by distance from fleet mean
            mean_hs = float(df["predicted_health_score"].mean())
            variances = {
                int(row["unit_id"]): abs(float(row["predicted_health_score"]) - mean_hs)
                for _, row in df.iterrows()
            }

        if not variances:
            return {
                "anomalous_vehicles": [],
                "their_variance_scores": {},
                "fleet_mean_variance": 0.0,
                "fleet_std_variance": 0.0,
                "note": "No timeline data available for variance computation.",
            }

        variance_series = pd.Series(variances)
        fleet_mean_var = float(variance_series.mean())
        fleet_std_var = float(variance_series.std(ddof=1))
        threshold_var = fleet_mean_var + 1.5 * fleet_std_var

        anomalous_ids = [
            uid for uid, var in variances.items() if var > threshold_var
        ]
        anomalous_vehicles = [f"Engine {uid}" for uid in sorted(anomalous_ids)]
        anomalous_scores = {
            f"Engine {uid}": round(variances[uid], 3) for uid in anomalous_ids
        }

        logger.info(
            "Anomaly hunt  fleet_mean_var=%.3f  fleet_std_var=%.3f  "
            "anomalous=%d",
            fleet_mean_var, fleet_std_var, len(anomalous_vehicles),
        )
        return {
            "anomalous_vehicles": anomalous_vehicles,
            "their_variance_scores": anomalous_scores,
            "fleet_mean_variance": round(fleet_mean_var, 3),
            "fleet_std_variance": round(fleet_std_var, 3),
            "anomaly_threshold": round(threshold_var, 3),
        }

    def _analyze_ranking_query(self, params: dict[str, Any]) -> dict[str, Any]:
        """Rank fleet vehicles by predicted health score (ascending = worst first).

        Args:
            params: May contain ``top_n`` (int); falls back to
                ``reasoning.default_top_n`` from config.

        Returns:
            Dict with ranked vehicle list, each entry including unit_id,
            health_score, risk_tier, and predicted_rul.
        """
        top_n: int = int(
            params.get("top_n") or self._rcfg.get("default_top_n", 5)
        )
        df = self._fleet_data.sort_values(
            "predicted_health_score", ascending=True
        ).head(top_n)

        ranked: list[dict[str, Any]] = [
            {
                "unit_id": f"Engine {int(row['unit_id'])}",
                "health_score": round(float(row["predicted_health_score"]), 1),
                "risk_tier": str(row["risk_tier"]),
                "predicted_rul": round(float(row["predicted_rul"]), 1),
            }
            for _, row in df.iterrows()
        ]

        logger.info("Ranking query  top_n=%d  worst=%s", top_n, ranked[0] if ranked else "N/A")
        return {
            "ranked_vehicles": ranked,
            "top_n": top_n,
            "worst_vehicle": ranked[0]["unit_id"] if ranked else "None",
            "worst_health_score": ranked[0]["health_score"] if ranked else 0.0,
        }

    def _analyze_trend_analysis(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compute fleet-average health score trend via linear regression.

        Uses the full health timeline DataFrame.  Falls back to a simplified
        analysis using the fleet snapshot if no timeline is available.

        Args:
            params: May contain ``time_window_cycles`` (int).

        Returns:
            Dict with trend direction, slope, start/end fleet averages, and
            the time window used.
        """
        time_window: int = int(params.get("time_window_cycles") or 50)

        if self._timeline_df is not None and not self._timeline_df.empty:
            fleet_avg = (
                self._timeline_df.groupby("cycle")["health_score"]
                .mean()
                .reset_index()
                .sort_values("cycle")
            )
            if len(fleet_avg) < 2:
                return self._trend_fallback(time_window)

            # Take the last `time_window` cycles
            fleet_avg = fleet_avg.tail(time_window).reset_index(drop=True)
            x = fleet_avg["cycle"].values.astype(float)
            y = fleet_avg["health_score"].values.astype(float)

            # Linear regression slope via numpy
            coeffs = np.polyfit(x, y, 1)
            slope = float(coeffs[0])

            fleet_avg_start = float(fleet_avg["health_score"].iloc[0])
            fleet_avg_end = float(fleet_avg["health_score"].iloc[-1])
            actual_window = int(fleet_avg["cycle"].iloc[-1]) - int(fleet_avg["cycle"].iloc[0])
        else:
            return self._trend_fallback(time_window)

        trend_direction = "improving" if slope > 0 else "declining"
        logger.info(
            "Trend analysis  slope=%.5f  direction=%s  window=%d",
            slope, trend_direction, actual_window,
        )
        return {
            "trend_direction": trend_direction,
            "slope": round(slope, 5),
            "fleet_avg_start": round(fleet_avg_start, 2),
            "fleet_avg_end": round(fleet_avg_end, 2),
            "time_window": time_window,
            "actual_cycles_analysed": actual_window,
        }

    def _trend_fallback(self, time_window: int) -> dict[str, Any]:
        """Fallback trend analysis using fleet snapshot health scores.

        Args:
            time_window: Number of cycles requested (informational only).

        Returns:
            Dict with best-effort trend metrics based on single-point scores.
        """
        df = self._fleet_data
        mean_hs = float(df["predicted_health_score"].mean())
        pct_declining = float(
            (df["risk_tier"].isin(["WARNING", "CRITICAL"])).mean() * 100
        )
        trend_direction = "declining" if pct_declining > 40 else "stable"
        return {
            "trend_direction": trend_direction,
            "slope": None,
            "fleet_avg_start": None,
            "fleet_avg_end": round(mean_hs, 2),
            "time_window": time_window,
            "note": "Full timeline not available; trend estimated from fleet snapshot.",
            "pct_vehicles_degraded": round(pct_declining, 1),
        }

    # ── Step 3: Synthesise verdict ────────────────────────────────────────────

    def _synthesize_verdict(
        self,
        question: str,
        query_type: str,
        results: dict[str, Any],
    ) -> tuple[str, str]:
        """Generate verdict_summary and recommended_action via Gemini.

        Falls back to :meth:`_template_verdict` if Gemini is unavailable
        or returns malformed JSON.

        Args:
            question: The original fleet manager question.
            query_type: Classified query type string.
            results: Raw computed results dict from :meth:`_run_analysis`.

        Returns:
            Tuple of ``(verdict_summary, recommended_action)``.
        """
        if self._gemini_available:
            try:
                import google.generativeai as genai  # type: ignore[import]

                synth_temp: float = float(
                    self._gcfg.get("synthesis_temperature", 0.3)
                )
                max_tokens: int = int(
                    self._gcfg.get("max_output_tokens", 500)
                )
                generation_config = genai.GenerationConfig(
                    temperature=synth_temp,
                    max_output_tokens=max_tokens,
                )
                results_json = json.dumps(
                    {k: v for k, v in results.items() if not isinstance(v, pd.DataFrame)},
                    default=str,
                    indent=2,
                )
                user_msg = _SYNTHESIS_USER_TEMPLATE.format(
                    question=question,
                    query_type=query_type,
                    results_json=results_json,
                )
                response = self._gemini_model.generate_content(
                    [_SYNTHESIS_SYSTEM_PROMPT, user_msg],
                    generation_config=generation_config,
                )
                raw_text: str = response.text.strip()
                parsed = self._safe_json_parse(raw_text)
                if (
                    parsed
                    and "verdict_summary" in parsed
                    and "recommended_action" in parsed
                ):
                    logger.info("Gemini synthesis successful  query_type=%s", query_type)
                    return (
                        str(parsed["verdict_summary"]),
                        str(parsed["recommended_action"]),
                    )
                logger.warning(
                    "Gemini synthesis returned unexpected JSON — "
                    "falling back to template. raw=%r",
                    raw_text[:200],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini synthesis failed: %s — using template", exc)

        return self._template_verdict(query_type, results)

    def _template_verdict(
        self, query_type: str, results: dict[str, Any]
    ) -> tuple[str, str]:
        """Build a verdict from f-string templates as the final fallback.

        Args:
            query_type: Classified query type string.
            results: Raw computed results dict.

        Returns:
            Tuple of ``(verdict_summary, recommended_action)``.
        """
        if query_type == "ranking_query":
            top = results.get("ranked_vehicles", [])
            top3 = [v["unit_id"] for v in top[:3]]
            worst = results.get("worst_vehicle", "N/A")
            hs = results.get("worst_health_score", 0.0)
            summary = (
                f"The {results.get('top_n', len(top))} most at-risk vehicles are "
                f"{', '.join(top3) if top3 else 'none identified'}. "
                f"{worst} has the lowest health score at {hs:.1f}/100."
            )
            action = f"Schedule immediate inspection and maintenance for {worst}."

        elif query_type == "threshold_query":
            count = results.get("count", 0)
            threshold = results.get("threshold_cycles", 30)
            critical = results.get("most_critical_vehicle", "N/A")
            avg_rul = results.get("avg_rul", 0.0)
            summary = (
                f"{count} vehicle(s) have predicted RUL below {threshold} cycles. "
                f"Average RUL across these vehicles is {avg_rul:.1f} cycles. "
                f"{critical} is the most urgent case."
            )
            action = (
                f"Initiate maintenance scheduling for all {count} vehicles immediately, "
                f"prioritising {critical}."
            )

        elif query_type == "fleet_comparison":
            faster = results.get("faster_group", "unknown")
            ga_rate = results.get("group_a_decline_rate", 0.0)
            gb_rate = results.get("group_b_decline_rate", 0.0)
            diff = results.get("difference", 0.0)
            summary = (
                f"{faster} vehicles are degrading faster. "
                f"Group A decline rate: {ga_rate:.4f}/cycle. "
                f"Group B decline rate: {gb_rate:.4f}/cycle "
                f"(difference: {diff:.4f}/cycle)."
            )
            action = (
                f"Increase maintenance frequency and inspection cadence for {faster} vehicles."
            )

        elif query_type == "anomaly_hunt":
            anom = results.get("anomalous_vehicles", [])
            count = len(anom)
            mean_var = results.get("fleet_mean_variance", 0.0)
            summary = (
                f"{count} vehicle(s) show statistically unusual health patterns "
                f"(variance > fleet mean of {mean_var:.3f}). "
                f"Flagged: {', '.join(anom[:5]) if anom else 'none'}."
            )
            action = (
                "Perform detailed diagnostic inspection on flagged vehicles to "
                "identify root cause of irregular health fluctuations."
            )

        elif query_type == "trend_analysis":
            direction = results.get("trend_direction", "unknown")
            slope = results.get("slope")
            window = results.get("time_window", 50)
            slope_str = f"{slope:.5f}/cycle" if slope is not None else "N/A"
            start = results.get("fleet_avg_start")
            end = results.get("fleet_avg_end", 0.0)
            start_str = f"{start:.2f}" if start is not None else "N/A"
            summary = (
                f"Fleet health is {direction} over the last {window} cycles "
                f"(slope: {slope_str}). "
                f"Fleet average health moved from {start_str} → {end:.2f}."
            )
            action = (
                "Increase fleet-wide preventive maintenance frequency."
                if direction == "declining"
                else "Current maintenance schedule is effective — continue monitoring."
            )

        else:
            summary = "The query could not be fully analysed with available data."
            action = "Please rephrase your question or check fleet data availability."

        return summary, action

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_json_parse(text: str) -> dict[str, Any] | None:
        """Parse JSON defensively, stripping markdown code fences if present.

        Args:
            text: Raw string that should contain a JSON object.

        Returns:
            Parsed dict, or ``None`` on any parsing error.
        """
        try:
            # Strip ```json ... ``` fences that Gemini sometimes adds
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _normalise_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
        """Ensure the parsed dict always contains the expected keys.

        Args:
            parsed: Dict returned by Gemini parser (may be incomplete).

        Returns:
            Dict guaranteed to have ``query_type``, ``params``,
            ``confidence``.
        """
        params = parsed.get("params") or {}
        default_params = {
            "threshold_cycles": None,
            "top_n": None,
            "time_window_cycles": None,
            "group_a_label": None,
            "group_b_label": None,
        }
        default_params.update(params)
        return {
            "query_type": parsed.get("query_type", "unknown"),
            "params": default_params,
            "confidence": float(parsed.get("confidence", 0.5)),
        }

    def _infer_confirmed(self, query_type: str, results: dict[str, Any]) -> bool:
        """Infer whether the hypothesis in the question is supported by data.

        Args:
            query_type: Classified query type.
            results: Raw analysis results.

        Returns:
            ``True`` if the analysis found supporting evidence.
        """
        if query_type == "threshold_query":
            return results.get("count", 0) > 0
        if query_type == "ranking_query":
            return len(results.get("ranked_vehicles", [])) > 0
        if query_type == "anomaly_hunt":
            return len(results.get("anomalous_vehicles", [])) > 0
        if query_type == "fleet_comparison":
            return results.get("difference", 0.0) > 0.001
        if query_type == "trend_analysis":
            return results.get("trend_direction") is not None
        return False

    def _extract_supporting_vehicles(
        self, query_type: str, results: dict[str, Any]
    ) -> list[str]:
        """Extract a flat list of vehicle ID strings from analysis results.

        Args:
            query_type: Classified query type.
            results: Raw analysis results.

        Returns:
            List of vehicle ID strings (e.g. ``["Engine 5", "Engine 12"]``).
        """
        if query_type == "threshold_query":
            return results.get("vehicles_needing_maintenance", [])
        if query_type == "ranking_query":
            return [v["unit_id"] for v in results.get("ranked_vehicles", [])]
        if query_type == "anomaly_hunt":
            return results.get("anomalous_vehicles", [])
        if query_type in ("fleet_comparison", "trend_analysis"):
            # No vehicle-level list — summarise at fleet level
            return []
        return []


# ── Standalone entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from src.utils.config_loader import load_config as _load_config
    from src.data_layer.loader import compute_rul, load_raw_data
    from src.data_layer.health_score import compute_health_scores, get_fleet_snapshot

    cfg = _load_config()
    rul_cap = cfg["data"]["rul_cap"]

    # ── Load Module 1 data (health timeline + snapshot) ───────────────────────
    df_train_raw, _, _ = load_raw_data()
    df_train = compute_rul(df_train_raw)
    df_scored = compute_health_scores(df_train, rul_cap)

    # ── Try to load Module 2 fleet predictions ────────────────────────────────
    fleet_pred_df = None
    try:
        from src.ml_layer.lstm_predictor import Predictor, build_sequences_with_unit_ids
        from src.data_layer.loader import drop_constant_sensors
        import joblib
        import numpy as _np

        df_train2, df_test, df_rul = load_raw_data()
        df_rul_idx = df_rul.reset_index(drop=True)
        df_rul_idx["unit_id"] = df_rul_idx.index + 1
        last_cycles_df = df_test.groupby("unit_id")["cycle"].max().reset_index()
        last_cycles_df = last_cycles_df.merge(
            df_rul_idx.rename(columns={"rul": "RUL"}), on="unit_id"
        )
        df_test = df_test.merge(
            last_cycles_df[["unit_id", "cycle", "RUL"]], on=["unit_id", "cycle"], how="left"
        )
        df_test["RUL"] = df_test.groupby("unit_id")["RUL"].transform(lambda s: s.bfill())
        df_test["RUL"] = df_test["RUL"].clip(upper=rul_cap).astype(_np.float32)
        df_test_clean = drop_constant_sensors(df_test)

        window_size = cfg["data"]["window_size"]
        X_te, _, unit_ids = build_sequences_with_unit_ids(df_test_clean, window_size)
        scaler = joblib.load(cfg["data"]["scaler_path"])
        n, ws, nf = X_te.shape
        X_te = scaler.transform(X_te.reshape(-1, nf)).reshape(n, ws, nf).astype("float32")
        last_cycles_arr = _np.array([
            df_test_clean[df_test_clean["unit_id"] == uid]
            .sort_values("cycle")["cycle"].iloc[-1]
            for uid in unit_ids
        ], dtype=_np.int64)

        predictor = Predictor(cfg)
        fleet_pred_df = predictor.predict_fleet(X_te, unit_ids, last_cycles_arr)
        print("✓  Module 2 fleet predictions loaded.\n")
    except Exception as e:
        print(f"⚠  Module 2 not available ({e}); using Module 1 snapshot as fleet_data.\n")
        snap = get_fleet_snapshot(df_scored)
        fleet_pred_df = snap.rename(columns={
            "health_score": "predicted_health_score",
            "rul_remaining": "predicted_rul",
            "current_cycle": "last_cycle",
        })

    # ── Initialise reasoner ───────────────────────────────────────────────────
    reasoner = HypothesisReasoner(cfg, fleet_pred_df, health_timeline_df=df_scored)

    questions = [
        "Which vehicles will need maintenance in the next 30 cycles?",
        "Show me the 5 most at-risk vehicles right now",
        "Are Group A vehicles degrading faster than Group B?",
        "Which vehicles show unusual health patterns?",
        "Is fleet health improving or declining over the last 50 cycles?",
    ]

    print("=" * 70)
    print("  TRACE — Module 3: Hypothesis Reasoner Demo")
    print("=" * 70)

    for q in questions:
        result = reasoner.analyze(q)
        backend = "Gemini" if result.used_gemini else "Rule-based / Template"
        print(f"\n❓ {q}")
        print(f"   Query type : {result.query_type}  (confidence={result.confidence:.2f})")
        print(f"   Backend    : {backend}")
        print(f"   Confirmed  : {result.confirmed}")
        print(f"   Verdict    : {result.verdict_summary}")
        print(f"   Action     : {result.recommended_action}")
        print("-" * 70)
