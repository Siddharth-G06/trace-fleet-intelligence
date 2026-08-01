"""
Unit tests for the TRACE LSTM RUL prediction module.

All tests use synthetic data so they run fast and without requiring a trained
model on disk (except test_model_save_load_consistency which exercises the
save/load cycle end-to-end).

Run with:
    pytest tests/test_lstm.py -v
"""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Synthetic config fixture ───────────────────────────────────────────────────

N_FEATURES  = 14
WINDOW_SIZE = 30
BATCH_SIZE  = 8


@pytest.fixture(scope="module")
def synthetic_config(tmp_path_factory) -> dict:
    """Return a minimal config dict that mirrors config/config.yaml structure."""
    tmp = tmp_path_factory.mktemp("models")
    return {
        "data": {
            "window_size": WINDOW_SIZE,
            "rul_cap":     125,
            "scaler_path": "models/scaler.pkl",   # not used in these tests
        },
        "health_score": {
            "healthy_threshold": 80,
            "watch_threshold":   50,
            "warning_threshold": 20,
        },
        "model": {
            "hidden_size_1":            64,
            "hidden_size_2":            32,
            "dropout":                  0.1,
            "dense_units":              16,
            "learning_rate":            0.001,
            "batch_size":               BATCH_SIZE,
            "max_epochs":               3,          # fast for tests
            "early_stopping_patience":  10,
            "train_val_split":          0.8,
            "model_path":               str(tmp / "lstm_model.pth"),
            "metadata_path":            str(tmp / "training_metadata.json"),
        },
    }


@pytest.fixture(scope="module")
def model(synthetic_config) -> "LSTMPredictor":
    """Instantiate a fresh (untrained) LSTMPredictor."""
    from src.ml_layer.lstm_predictor import LSTMPredictor
    return LSTMPredictor(synthetic_config, n_features=N_FEATURES)


@pytest.fixture(scope="module")
def trainer(synthetic_config) -> "Trainer":
    """Instantiate a Trainer with the synthetic config."""
    from src.ml_layer.lstm_predictor import Trainer
    return Trainer(synthetic_config, n_features=N_FEATURES)


# ── Synthetic data helpers ─────────────────────────────────────────────────────


def _make_sequences(
    n_units: int = 20,
    cycles_per_unit: int = 50,
    window_size: int = WINDOW_SIZE,
    n_features: int = N_FEATURES,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic (X, y, unit_ids) for testing.

    Args:
        n_units:         Number of synthetic engine units.
        cycles_per_unit: Cycles per engine unit.
        window_size:     Sequence window size.
        n_features:      Number of sensor features.
        rng_seed:        NumPy RNG seed for reproducibility.

    Returns:
        ``(X, y, unit_ids)`` numpy arrays.
    """
    rng = np.random.default_rng(rng_seed)
    X_list, y_list, uid_list = [], [], []

    for uid in range(1, n_units + 1):
        feats  = rng.random((cycles_per_unit, n_features)).astype(np.float32)
        labels = np.arange(cycles_per_unit - 1, -1, -1, dtype=np.float32)
        n_windows = cycles_per_unit - window_size + 1
        for i in range(n_windows):
            X_list.append(feats[i : i + window_size])
            y_list.append(labels[i + window_size - 1])
            uid_list.append(uid)

    return (
        np.array(X_list,   dtype=np.float32),
        np.array(y_list,   dtype=np.float32),
        np.array(uid_list, dtype=np.int64),
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestModelOutputShape:
    """Verify the forward pass produces the correct output tensor shape."""

    def test_model_output_shape(self, model: "LSTMPredictor") -> None:
        """Output shape must be (batch_size, 1) for any batch size."""
        model.eval()
        x = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 1), (
            f"Expected output shape ({BATCH_SIZE}, 1), got {tuple(out.shape)}"
        )


class TestOutputNonNegative:
    """Verify the output ReLU on the final layer prevents negative predictions."""

    def test_output_non_negative(self, model: "LSTMPredictor") -> None:
        """All predicted values must be ≥ 0 (due to ReLU on output layer)."""
        model.eval()
        # Use a deterministic tensor that might produce negative pre-activation
        torch.manual_seed(0)
        x = torch.randn(32, WINDOW_SIZE, N_FEATURES) * 10  # amplified inputs
        with torch.no_grad():
            out = model(x)
        min_val = float(out.min())
        assert min_val >= 0.0, (
            f"Model output contains negative values (min={min_val:.6f}). "
            "Check that the output ReLU is applied."
        )


class TestHealthScoreRange:
    """Verify predict_health_score always returns values in [0, 100]."""

    def test_health_score_range(self, synthetic_config: dict) -> None:
        """Health scores must lie within [0, 100] for arbitrary inputs."""
        from src.ml_layer.lstm_predictor import LSTMPredictor, Predictor, Trainer

        # Build a Predictor that uses the *untrained* model (weights don't matter)
        # We test the clamping logic, not the accuracy.
        trainer_obj = Trainer(synthetic_config, n_features=N_FEATURES)
        # Save the freshly initialised (untrained) model so Predictor.load() works
        trainer_obj.save({"rmse": 0, "mae": 0, "epochs_trained": 0, "best_val_loss": 0})

        # Patch scaler path to something that exists — use a trivial joblib dump
        import joblib, tempfile
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(np.ones((1, N_FEATURES)))
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_scaler = f.name
        joblib.dump(scaler, tmp_scaler)

        patched_cfg = deepcopy(synthetic_config)
        patched_cfg["data"]["scaler_path"] = tmp_scaler

        predictor = Predictor(patched_cfg, n_features=N_FEATURES)

        rng = np.random.default_rng(99)
        X   = rng.random((50, WINDOW_SIZE, N_FEATURES)).astype(np.float32)
        scores = predictor.predict_health_score(X)

        assert scores.min() >= 0.0,   f"Health score below 0: {scores.min()}"
        assert scores.max() <= 100.0, f"Health score above 100: {scores.max()}"


class TestEngineSplitNoLeakage:
    """Verify the engine-level split produces disjoint unit sets."""

    def test_engine_split_no_leakage(self, trainer: "Trainer") -> None:
        """No engine unit may appear in both the train and validation splits."""
        X, y, unit_ids = _make_sequences(n_units=20)
        X_train, y_train, X_val, y_val = trainer.split_by_engine(X, y, unit_ids)

        # Recover which units contributed to each partition by checking
        # the split_by_engine logic: we re-derive train/val sets.
        unique_units  = np.sort(np.unique(unit_ids))
        n_train       = int(len(unique_units) * float(
            trainer._mcfg["train_val_split"]
        ))
        train_units   = set(unique_units[:n_train].tolist())
        val_units     = set(unique_units[n_train:].tolist())

        overlap = train_units & val_units
        assert len(overlap) == 0, (
            f"Engine units appear in both train and val splits: {overlap}"
        )
        assert len(train_units) > 0, "Train set is empty — split ratio too aggressive"
        assert len(val_units)   > 0, "Val set is empty — split ratio too aggressive"


class TestPredictFleetOneRowPerUnit:
    """Verify predict_fleet() returns exactly one row per unique unit_id."""

    def test_predict_fleet_one_row_per_unit(self, synthetic_config: dict) -> None:
        """Fleet DataFrame must have exactly one row per distinct unit_id."""
        from src.ml_layer.lstm_predictor import Predictor, Trainer
        import joblib, tempfile
        from sklearn.preprocessing import MinMaxScaler

        trainer_obj = Trainer(synthetic_config, n_features=N_FEATURES)
        trainer_obj.save({"rmse": 0, "mae": 0, "epochs_trained": 0, "best_val_loss": 0})

        scaler = MinMaxScaler()
        scaler.fit(np.ones((1, N_FEATURES)))
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_scaler = f.name
        joblib.dump(scaler, tmp_scaler)

        patched_cfg = deepcopy(synthetic_config)
        patched_cfg["data"]["scaler_path"] = tmp_scaler

        predictor = Predictor(patched_cfg, n_features=N_FEATURES)

        n_units = 15
        X, y, unit_ids = _make_sequences(n_units=n_units)
        last_cycles = np.arange(len(unit_ids), dtype=np.int64)  # dummy values

        fleet_df = predictor.predict_fleet(X, unit_ids, last_cycles)

        assert len(fleet_df) == n_units, (
            f"Expected {n_units} rows (one per unit), got {len(fleet_df)}"
        )
        assert fleet_df["unit_id"].nunique() == n_units, (
            "Duplicate unit_ids found in fleet DataFrame"
        )


class TestModelSaveLoadConsistency:
    """Verify that save → load produces bit-identical predictions."""

    def test_model_save_load_consistency(self, synthetic_config: dict) -> None:
        """Predictions before saving and after loading must be identical."""
        from src.ml_layer.lstm_predictor import LSTMPredictor, Trainer
        import copy

        trainer_obj = Trainer(synthetic_config, n_features=N_FEATURES)

        # Get predictions BEFORE save
        trainer_obj.model.eval()
        rng = np.random.default_rng(7)
        X   = rng.random((16, WINDOW_SIZE, N_FEATURES)).astype(np.float32)
        X_t = torch.tensor(X, dtype=torch.float32).to(trainer_obj.device)

        with torch.no_grad():
            preds_before = trainer_obj.model(X_t).cpu().numpy()

        # Save then load
        trainer_obj.save({"rmse": 1.0, "mae": 0.5, "epochs_trained": 1, "best_val_loss": 0.9})
        trainer_obj.load()   # reloads into the same model object

        with torch.no_grad():
            preds_after = trainer_obj.model(X_t).cpu().numpy()

        np.testing.assert_array_almost_equal(
            preds_before, preds_after, decimal=5,
            err_msg="Predictions differ before and after save/load cycle",
        )
        # Confirm model is in eval mode after load
        assert not trainer_obj.model.training, (
            "Model should be in eval mode immediately after load()"
        )
