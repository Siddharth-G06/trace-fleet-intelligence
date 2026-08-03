"""
TRACE — Module 3: Unit tests for HypothesisReasoner.

Tests cover:
1. Rule-based parser — threshold_query
2. Rule-based parser — ranking_query
3. Malformed Gemini JSON handled gracefully (no exception)
4. Threshold analysis returns exactly the correct vehicles
5. VerdictResult has no None fields after analyze()
6. Ranking analysis returns vehicles sorted ascending by health_score
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning_layer.hypothesis_reasoner import HypothesisReasoner, VerdictResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_config() -> dict:
    """Minimal configuration dict sufficient for HypothesisReasoner."""
    return {
        "gemini": {
            "api_key": "",
            "model": "gemini-1.5-flash",
            "parser_temperature": 0.0,
            "synthesis_temperature": 0.3,
            "max_output_tokens": 500,
        },
        "reasoning": {
            "supported_query_types": [
                "fleet_comparison",
                "threshold_query",
                "anomaly_hunt",
                "ranking_query",
                "trend_analysis",
            ],
            "default_top_n": 5,
            "default_threshold_cycles": 30,
            "group_a_units": [1, 50],
            "group_b_units": [51, 100],
            "health_decline_window": 20,
        },
        "health_score": {
            "healthy_threshold": 80,
            "watch_threshold": 50,
            "warning_threshold": 20,
        },
    }


@pytest.fixture()
def mock_fleet_data() -> pd.DataFrame:
    """Small synthetic fleet DataFrame with 10 engines.

    Engines 1, 5, 9 have predicted_rul < 30 (should be caught by threshold).
    Engines are sorted ascending by predicted_health_score in this order:
    engine 9 (hs=5), 5 (hs=10), 1 (hs=22), 2 (hs=35), 3 (hs=50), …
    """
    rows = [
        # unit_id  rul   health  tier        cycle
        (1,        20.0,  22.0, "WARNING",   100),
        (2,        45.0,  35.0, "WARNING",    90),
        (3,        60.0,  50.0, "WATCH",      80),
        (4,        80.0,  65.0, "WATCH",      70),
        (5,        12.0,  10.0, "CRITICAL",  110),
        (6,        90.0,  72.0, "WATCH",      60),
        (7,       100.0,  82.0, "HEALTHY",    50),
        (8,       110.0,  88.0, "HEALTHY",    40),
        (9,         5.0,   5.0, "CRITICAL",  120),
        (10,      105.0,  85.0, "HEALTHY",    35),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "unit_id",
            "predicted_rul",
            "predicted_health_score",
            "risk_tier",
            "last_cycle",
        ],
    )


@pytest.fixture()
def reasoner_no_gemini(minimal_config, mock_fleet_data) -> HypothesisReasoner:
    """HypothesisReasoner with Gemini disabled (no API key)."""
    return HypothesisReasoner(minimal_config, mock_fleet_data)


# ── Test 1: Rule-based parser — threshold_query ───────────────────────────────

def test_rule_based_threshold_query(reasoner_no_gemini):
    """Rule-based parser should detect threshold_query from maintenance keywords."""
    result = reasoner_no_gemini._rule_based_parser(
        "which vehicles need maintenance in 30 cycles"
    )
    assert result["query_type"] == "threshold_query", (
        f"Expected 'threshold_query', got {result['query_type']!r}"
    )
    assert result["confidence"] > 0.5


# ── Test 2: Rule-based parser — ranking_query ─────────────────────────────────

def test_rule_based_ranking_query(reasoner_no_gemini):
    """Rule-based parser should detect ranking_query from 'worst' keyword."""
    result = reasoner_no_gemini._rule_based_parser("show worst vehicles")
    assert result["query_type"] == "ranking_query", (
        f"Expected 'ranking_query', got {result['query_type']!r}"
    )
    assert result["confidence"] > 0.5


# ── Test 3: Malformed Gemini JSON handled gracefully ─────────────────────────

def test_malformed_json_handled(minimal_config, mock_fleet_data):
    """When Gemini returns invalid JSON, no exception should be raised and
    the parser should fall back to rule-based, returning a valid dict."""
    # Build a mock Gemini response that returns garbage text
    mock_response = MagicMock()
    mock_response.text = "THIS IS NOT JSON {{{{!!"

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    reasoner = HypothesisReasoner(minimal_config, mock_fleet_data)
    # Manually inject the mock model and mark Gemini as available
    reasoner._gemini_model = mock_model
    reasoner._gemini_available = True

    # Should not raise; should fall back to rule-based parser
    result = reasoner._parse_question("which vehicles need maintenance")
    assert isinstance(result, dict)
    assert "query_type" in result
    # Rule-based parser is the fallback, so query_type must be a valid value
    assert result["query_type"] in {
        "fleet_comparison",
        "threshold_query",
        "anomaly_hunt",
        "ranking_query",
        "trend_analysis",
        "unknown",
    }


# ── Test 4: Threshold analysis returns exactly the right vehicles ─────────────

def test_threshold_analysis_returns_correct_vehicles(mock_fleet_data, minimal_config):
    """Inject fleet_data where engines 1, 5, 9 have rul < 30.
    _analyze_threshold_query should return exactly those 3 vehicles."""
    reasoner = HypothesisReasoner(minimal_config, mock_fleet_data)
    results = reasoner._analyze_threshold_query({"threshold_cycles": 30})

    returned_vehicles = set(results["vehicles_needing_maintenance"])
    expected_vehicles = {"Engine 1", "Engine 5", "Engine 9"}

    assert returned_vehicles == expected_vehicles, (
        f"Expected {expected_vehicles}, got {returned_vehicles}"
    )
    assert results["count"] == 3
    assert results["most_critical_vehicle"] == "Engine 9"  # lowest RUL = 5.0


# ── Test 5: VerdictResult has all fields populated (no None) ─────────────────

def test_verdict_result_fields_complete(reasoner_no_gemini):
    """analyze() must always return a VerdictResult with all fields non-None."""
    result = reasoner_no_gemini.analyze("Which vehicles need maintenance soon?")

    assert isinstance(result, VerdictResult)
    # Every field must be non-None
    for field_name, value in result.__dict__.items():
        assert value is not None, (
            f"VerdictResult.{field_name} should not be None"
        )
    # Specific type checks
    assert isinstance(result.original_question, str) and result.original_question
    assert isinstance(result.query_type, str) and result.query_type
    assert isinstance(result.confidence, float)
    assert isinstance(result.verdict_summary, str) and result.verdict_summary
    assert isinstance(result.supporting_vehicles, list)
    assert isinstance(result.key_metrics, dict)
    assert isinstance(result.recommended_action, str) and result.recommended_action
    assert isinstance(result.computed_at, datetime)


# ── Test 6: Ranking returns vehicles sorted ascending by health_score ─────────

def test_ranking_returns_sorted_by_health(mock_fleet_data, minimal_config):
    """_analyze_ranking_query must return vehicles sorted ascending by health_score."""
    reasoner = HypothesisReasoner(minimal_config, mock_fleet_data)
    results = reasoner._analyze_ranking_query({"top_n": 5})

    ranked = results["ranked_vehicles"]
    assert len(ranked) == 5

    scores = [v["health_score"] for v in ranked]
    assert scores == sorted(scores), (
        f"Vehicles not sorted ascending: {scores}"
    )
    # The worst vehicle must be first
    assert ranked[0]["unit_id"] == "Engine 9"  # health_score = 5.0
    assert ranked[0]["health_score"] == 5.0
