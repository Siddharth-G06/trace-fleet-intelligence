"""
Unit tests for the TRACE data foundation layer.

Covers RUL computation, sensor dropping, sequence building, normalization,
health scoring, and fleet snapshot generation.

Run with:
    pytest tests/test_data_layer.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def raw_train() -> pd.DataFrame:
    """Load the raw training DataFrame once per test session."""
    from src.data_layer.loader import load_raw_data

    df_train, _, _ = load_raw_data()
    return df_train


@pytest.fixture(scope="module")
def train_with_rul(raw_train: pd.DataFrame) -> pd.DataFrame:
    """Training DataFrame with RUL column added."""
    from src.data_layer.loader import compute_rul

    return compute_rul(raw_train)


@pytest.fixture(scope="module")
def clean_train(train_with_rul: pd.DataFrame) -> pd.DataFrame:
    """Training DataFrame with constant sensors removed."""
    from src.data_layer.loader import drop_constant_sensors

    return drop_constant_sensors(train_with_rul)


@pytest.fixture(scope="module")
def pipeline_result() -> dict:
    """Full pipeline result dict (expensive — cached across tests)."""
    from src.data_layer.loader import run_pipeline

    return run_pipeline()


@pytest.fixture(scope="module")
def fleet_snapshot_data(train_with_rul: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(df_scored, fleet_snapshot) for health-score tests."""
    from src.data_layer.health_score import compute_health_scores, get_fleet_snapshot
    from src.utils.config_loader import load_config

    rul_cap: int = load_config()["data"]["rul_cap"]
    df_scored = compute_health_scores(train_with_rul, rul_cap)
    snapshot = get_fleet_snapshot(df_scored)
    return df_scored, snapshot


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRulComputation:
    """Validate RUL label correctness."""

    def test_rul_computation_correct(self, train_with_rul: pd.DataFrame) -> None:
        """For every unit, the RUL at the last cycle must be exactly 0."""
        # Find the row index of the max cycle per unit, then read its RUL
        idx_last = train_with_rul.groupby("unit_id")["cycle"].idxmax()
        last_cycle_rul = train_with_rul.loc[idx_last, "RUL"]
        # The last cycle has max_cycle - max_cycle = 0, before the clip.
        # clip(upper=rul_cap) does NOT affect 0, so all values must be 0.
        assert (last_cycle_rul == 0).all(), (
            f"Expected RUL=0 at last cycle for all units, "
            f"but found non-zero values: {last_cycle_rul[last_cycle_rul != 0]}"
        )

    def test_rul_cap_applied(self, train_with_rul: pd.DataFrame) -> None:
        """No RUL value should exceed the configured rul_cap."""
        from src.utils.config_loader import load_config

        rul_cap: int = load_config()["data"]["rul_cap"]
        max_rul = train_with_rul["RUL"].max()
        assert max_rul <= rul_cap, (
            f"RUL cap violation: max RUL={max_rul} exceeds rul_cap={rul_cap}"
        )


class TestSensorDropping:
    """Validate sensor-dropping behaviour."""

    def test_sensors_dropped(self, pipeline_result: dict) -> None:
        """Configured sensors_to_drop must not appear in feature_names."""
        from src.utils.config_loader import load_config

        to_drop: list[str] = load_config()["data"]["sensors_to_drop"]
        feature_names: list[str] = pipeline_result["feature_names"]

        for sensor in to_drop:
            assert sensor not in feature_names, (
                f"Sensor '{sensor}' should have been dropped but is still in feature_names."
            )


class TestSequenceBuilding:
    """Validate sliding-window sequence construction."""

    def test_sequence_shape(self, pipeline_result: dict) -> None:
        """X_train must have exactly 3 dimensions: (n, window_size, n_features)."""
        from src.utils.config_loader import load_config

        window_size: int = load_config()["data"]["window_size"]
        X_train: np.ndarray = pipeline_result["X_train"]

        assert X_train.ndim == 3, (
            f"Expected X_train.ndim=3, got {X_train.ndim}"
        )
        assert X_train.shape[1] == window_size, (
            f"Expected sequence length={window_size}, got {X_train.shape[1]}"
        )


class TestNormalization:
    """Validate MinMaxScaler normalization."""

    def test_scaler_fit_on_train_only(self, pipeline_result: dict) -> None:
        """All values in X_train_scaled must lie within [0, 1]."""
        X_train: np.ndarray = pipeline_result["X_train"]
        assert float(X_train.min()) >= -1e-6, (
            f"X_train min={X_train.min():.6f} is below 0 after scaling."
        )
        assert float(X_train.max()) <= 1.0 + 1e-6, (
            f"X_train max={X_train.max():.6f} exceeds 1 after scaling."
        )


class TestHealthScores:
    """Validate health score and risk tier computation."""

    def test_health_score_range(
        self, fleet_snapshot_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """All health_score values must be between 0 and 100 (inclusive)."""
        df_scored, _ = fleet_snapshot_data
        scores = df_scored["health_score"]
        assert scores.min() >= 0.0, f"Health score below 0: {scores.min()}"
        assert scores.max() <= 100.0, f"Health score above 100: {scores.max()}"


class TestFleetSnapshot:
    """Validate fleet snapshot structure."""

    def test_fleet_snapshot_one_row_per_unit(
        self, fleet_snapshot_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Fleet snapshot must have exactly one row per unique unit_id."""
        df_scored, snapshot = fleet_snapshot_data
        n_units = df_scored["unit_id"].nunique()
        assert len(snapshot) == n_units, (
            f"Expected {n_units} rows in fleet snapshot, got {len(snapshot)}"
        )

    def test_fleet_snapshot_sorted_by_health_score(
        self, fleet_snapshot_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Fleet snapshot rows must be sorted health_score ascending."""
        _, snapshot = fleet_snapshot_data
        scores = snapshot["health_score"].tolist()
        assert scores == sorted(scores), "Fleet snapshot is not sorted by health_score ascending."
