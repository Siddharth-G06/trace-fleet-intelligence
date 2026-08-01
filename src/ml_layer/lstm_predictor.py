"""
LSTM-based Remaining Useful Life predictor for the TRACE fleet intelligence system.

This module provides three components:

* :class:`LSTMPredictor` — a two-layer LSTM regression model built with PyTorch.
* :class:`Trainer`       — explicit training loop with early stopping, engine-level
  train/val split, and artifact persistence.
* :class:`Predictor`    — stateless inference wrapper that converts RUL predictions
  to Vehicle Health Scores and fleet-level DataFrames.

All hyperparameters are read from ``config/config.yaml`` under the ``model`` key.
No magic numbers are hard-coded.  All logging is routed through
:func:`~src.utils.logger.get_logger`; no ``print`` statements are used inside
classes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not already exist.

    Args:
        path: File path whose parent directory should be created.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_sequences_with_unit_ids(
    df: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences and record the unit_id for each window.

    Mirrors :func:`~src.data_layer.loader.build_sequences` but additionally
    returns the ``unit_id`` of each window so downstream code can perform
    engine-level splitting and per-engine fleet predictions.

    Args:
        df: DataFrame containing sensor columns and a ``RUL`` column.
            Must also contain ``unit_id`` and ``cycle``.
        window_size: Number of cycles per sequence window.

    Returns:
        A three-tuple ``(X, y, unit_ids)`` where:

        * ``X``        — shape ``(n_windows, window_size, n_features)``
        * ``y``        — shape ``(n_windows,)``  — RUL at last cycle in window
        * ``unit_ids`` — shape ``(n_windows,)``  — unit_id owning the window
    """
    feature_cols = [
        c for c in df.columns if c not in ("unit_id", "cycle", "RUL")
    ]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    uid_list: list[int] = []

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle")
        features = group[feature_cols].values          # (n_cycles, n_features)
        labels   = group["RUL"].values                 # (n_cycles,)
        n_windows = len(features) - window_size + 1

        for i in range(n_windows):
            X_list.append(features[i : i + window_size])
            y_list.append(labels[i + window_size - 1])
            uid_list.append(int(unit_id))

    X        = np.array(X_list,   dtype=np.float32)
    y        = np.array(y_list,   dtype=np.float32)
    unit_ids = np.array(uid_list, dtype=np.int64)

    logger.info(
        "Sequences built  X.shape=%s  y.shape=%s  window_size=%d  units=%d",
        X.shape, y.shape, window_size, len(np.unique(unit_ids)),
    )
    return X, y, unit_ids


# ── Model ──────────────────────────────────────────────────────────────────────


class LSTMPredictor(nn.Module):
    """Two-layer LSTM regression model for RUL prediction.

    Architecture::

        Input  (batch, seq_len, n_features)
          → LSTM-1 (n_features → hidden_1, batch_first=True)
          → Dropout-1
          → LSTM-2 (hidden_1 → hidden_2, batch_first=True)
          → Dropout-2
          → Linear (hidden_2 → dense_units) + ReLU
          → Linear (dense_units → 1) + ReLU  [enforces non-negative RUL]
        Output (batch, 1)

    All sizes are read from the ``model`` section of ``config/config.yaml``.

    Args:
        config: The full parsed configuration dict (top-level).
        n_features: Number of input sensor features.  Defaults to 17
            (the 17 features — 14 sensors + 3 op_settings — kept after
            dropping the 7 low-variance sensors from NASA CMAPSS FD001).
    """

    def __init__(self, config: dict[str, Any], n_features: int = 17) -> None:
        super().__init__()
        m = config["model"]
        h1: int  = int(m["hidden_size_1"])
        h2: int  = int(m["hidden_size_2"])
        d:  float = float(m["dropout"])
        du: int  = int(m["dense_units"])

        self.lstm1    = nn.LSTM(n_features, h1, batch_first=True)
        self.drop1    = nn.Dropout(d)
        self.lstm2    = nn.LSTM(h1, h2, batch_first=True)
        self.drop2    = nn.Dropout(d)
        self.dense    = nn.Linear(h2, du)
        self.relu     = nn.ReLU()
        self.output   = nn.Linear(du, 1)
        self.out_relu = nn.ReLU()

        logger.info(
            "LSTMPredictor built  n_features=%d  h1=%d  h2=%d  "
            "dense_units=%d  dropout=%.2f",
            n_features, h1, h2, du, d,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Output tensor of shape ``(batch, 1)`` representing predicted RUL.
        """
        # LSTM-1: take only the final hidden state → (batch, hidden_1)
        out1, _ = self.lstm1(x)
        out1    = self.drop1(out1[:, -1, :])           # last time-step

        # LSTM-2 expects a sequence; unsqueeze back to (batch, 1, hidden_1)
        out2, _ = self.lstm2(out1.unsqueeze(1))
        out2    = self.drop2(out2[:, -1, :])           # last time-step

        z = self.relu(self.dense(out2))                # (batch, dense_units)
        return self.out_relu(self.output(z))           # (batch, 1)


# ── Trainer ────────────────────────────────────────────────────────────────────


class Trainer:
    """Manages model instantiation, training, evaluation, and persistence.

    Args:
        config: The full parsed configuration dict (top-level).
        n_features: Number of input sensor features.  Defaults to 17.
    """

    def __init__(self, config: dict[str, Any], n_features: int = 17) -> None:
        self._cfg  = config
        self._mcfg = config["model"]
        self._n_features = n_features

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer initialised  device=%s", self.device)

        self.model     = LSTMPredictor(config, n_features).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self._mcfg["learning_rate"]),
        )
        self.criterion = nn.MSELoss()

    # ── Data splitting ─────────────────────────────────────────────────────────

    def split_by_engine(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unit_ids: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split data 80/20 by unique engine unit — no row-level leakage.

        The set of unique ``unit_id`` values is sorted deterministically and
        sliced at 80 %.  All windows belonging to train-set engines form the
        training partition; all windows belonging to the remaining engines
        form the validation partition.

        Args:
            X:        Sequences array of shape ``(n_windows, seq_len, n_feat)``.
            y:        RUL labels array of shape ``(n_windows,)``.
            unit_ids: Engine unit for each window, shape ``(n_windows,)``.

        Returns:
            ``(X_train, y_train, X_val, y_val)`` as ``torch.Tensor`` on
            :attr:`device`.
        """
        split_ratio: float = float(self._mcfg["train_val_split"])

        unique_units = np.sort(np.unique(unit_ids))
        n_train_units = int(len(unique_units) * split_ratio)
        train_units   = set(unique_units[:n_train_units])
        val_units     = set(unique_units[n_train_units:])

        train_mask = np.isin(unit_ids, list(train_units))
        val_mask   = np.isin(unit_ids, list(val_units))

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float32).to(self.device)

        X_train = _to_tensor(X[train_mask])
        y_train = _to_tensor(y[train_mask]).unsqueeze(1)
        X_val   = _to_tensor(X[val_mask])
        y_val   = _to_tensor(y[val_mask]).unsqueeze(1)

        logger.info(
            "Engine split  train_units=%d  val_units=%d  "
            "train_windows=%d  val_windows=%d",
            len(train_units), len(val_units),
            X_train.shape[0], X_val.shape[0],
        )
        return X_train, y_train, X_val, y_val

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unit_ids: np.ndarray,
    ) -> dict[str, Any]:
        """Run the full training loop with early stopping.

        Steps performed:

        1. :meth:`split_by_engine` to get train / val tensors.
        2. Wrap in :class:`~torch.utils.data.DataLoader` objects.
        3. Loop over epochs:

           * ``model.train()`` + forward + backward + optimizer step.
           * ``model.eval()`` + ``torch.no_grad()`` for validation loss.
           * Early stopping: saves best ``state_dict`` and counts patience.

        4. Restore best model weights after training.

        Args:
            X:        Sequences array ``(n_windows, seq_len, n_features)``.
            y:        RUL labels ``(n_windows,)``.
            unit_ids: Engine unit per window ``(n_windows,)``.

        Returns:
            History dict::

                {
                    "train_losses":  list[float],
                    "val_losses":    list[float],
                    "best_val_loss": float,
                    "epochs_trained": int,
                }
        """
        batch_size: int = int(self._mcfg["batch_size"])
        max_epochs: int = int(self._mcfg["max_epochs"])
        patience:   int = int(self._mcfg["early_stopping_patience"])

        X_train, y_train, X_val, y_val = self.split_by_engine(X, y, unit_ids)

        train_ds     = TensorDataset(X_train, y_train)
        val_ds       = TensorDataset(X_val,   y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        train_losses: list[float] = []
        val_losses:   list[float] = []
        best_val_loss   = float("inf")
        best_state_dict = None
        patience_counter = 0
        epochs_trained   = 0

        logger.info(
            "Training START  max_epochs=%d  batch_size=%d  patience=%d",
            max_epochs, batch_size, patience,
        )

        for epoch in range(1, max_epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss  = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item() * len(X_batch)

            epoch_train_loss /= len(train_ds)

            # ── Validate ───────────────────────────────────────────────────────
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    preds = self.model(X_batch)
                    loss  = self.criterion(preds, y_batch)
                    epoch_val_loss += loss.item() * len(X_batch)
            epoch_val_loss /= len(val_ds)

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            epochs_trained = epoch

            logger.info(
                "Epoch %03d/%03d  train_loss=%.4f  val_loss=%.4f",
                epoch, max_epochs, epoch_train_loss, epoch_val_loss,
            )

            # ── Early stopping ─────────────────────────────────────────────────
            if epoch_val_loss < best_val_loss:
                best_val_loss    = epoch_val_loss
                best_state_dict  = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d  best_val_loss=%.4f",
                        epoch, best_val_loss,
                    )
                    break

        # Restore best weights
        if best_state_dict is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state_dict.items()}
            )
        self.model.eval()
        logger.info(
            "Training COMPLETE  epochs_trained=%d  best_val_loss=%.4f",
            epochs_trained, best_val_loss,
        )

        return {
            "train_losses":   train_losses,
            "val_losses":     val_losses,
            "best_val_loss":  best_val_loss,
            "epochs_trained": epochs_trained,
        }

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Compute RMSE and MAE on a held-out test set.

        Args:
            X_test: Test sequences ``(n_test, seq_len, n_features)``.
            y_test: Ground-truth RUL labels ``(n_test,)``.

        Returns:
            ``{"rmse": float, "mae": float}``
        """
        self.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(X_t).squeeze(1).cpu().numpy()

        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        mae  = float(np.mean(np.abs(preds - y_test)))

        logger.info("Evaluation  RMSE=%.4f  MAE=%.4f", rmse, mae)
        return {"rmse": rmse, "mae": mae}

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, metrics: dict[str, Any]) -> None:
        """Save model weights and training metadata to disk.

        * Saves ``state_dict`` to the path specified by ``model.model_path``.
        * Saves a JSON metadata file to ``model.metadata_path`` containing
          RMSE, MAE, epoch count, loss history, architecture info, and a UTC
          training timestamp.

        Args:
            metrics: Dict containing at minimum keys ``"rmse"``, ``"mae"``,
                ``"epochs_trained"``, and ``"best_val_loss"``.
        """
        model_path    = self._mcfg["model_path"]
        metadata_path = self._mcfg["metadata_path"]
        _ensure_dir(model_path)
        _ensure_dir(metadata_path)

        torch.save(self.model.state_dict(), model_path)
        logger.info("Model saved to '%s'", model_path)

        cfg_data = self._cfg["data"]
        metadata: dict[str, Any] = {
            "rmse":          metrics.get("rmse"),
            "mae":           metrics.get("mae"),
            "epochs_trained": metrics.get("epochs_trained"),
            "best_val_loss": metrics.get("best_val_loss"),
            "window_size":   cfg_data["window_size"],
            "n_features":    self._n_features,
            "hidden_size_1": self._mcfg["hidden_size_1"],
            "hidden_size_2": self._mcfg["hidden_size_2"],
            "dense_units":   self._mcfg["dense_units"],
            "model_path":    model_path,
            "trained_at":    datetime.now(timezone.utc).isoformat(),
        }

        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Metadata saved to '%s'", metadata_path)

    def load(self) -> None:
        """Load model weights from disk and switch the model to eval mode.

        Reads the path from ``config.model.model_path``.  Immediately calls
        ``model.eval()`` after loading so the model is ready for inference.

        Raises:
            FileNotFoundError: If the model file does not exist at the
                configured path.
        """
        model_path = self._mcfg["model_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'. "
                "Run training first (python -m src.ml_layer.lstm_predictor)."
            )
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("Model loaded from '%s'  (eval mode)", model_path)


# ── Predictor ──────────────────────────────────────────────────────────────────


class Predictor:
    """Stateless inference wrapper for the trained LSTM RUL model.

    Loads the persisted model and MinMaxScaler on construction; all
    ``predict_*`` methods are stateless after that point and safe to use
    from a cached ``st.cache_resource`` context.

    Args:
        config: The full parsed configuration dict (top-level).
        n_features: Number of input sensor features.  Defaults to 17.
    """

    def __init__(self, config: dict[str, Any], n_features: int = 17) -> None:
        self._cfg      = config
        self._mcfg     = config["model"]
        self._dcfg     = config["data"]
        self._rul_cap: float = float(config["data"]["rul_cap"])

        self._trainer = Trainer(config, n_features)
        self._trainer.load()

        scaler_path = self._dcfg["scaler_path"]
        if not Path(scaler_path).exists():
            raise FileNotFoundError(
                f"Scaler not found: '{scaler_path}'. "
                "Run the data pipeline first."
            )
        self._scaler = joblib.load(scaler_path)
        logger.info("Predictor ready  rul_cap=%.0f", self._rul_cap)

    @property
    def model(self) -> LSTMPredictor:
        """The underlying :class:`LSTMPredictor` module."""
        return self._trainer.model

    @property
    def device(self) -> torch.device:
        """The compute device used by the model."""
        return self._trainer.device

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        """Run inference and return clipped RUL predictions.

        Args:
            X: Sequences array of shape ``(n_windows, seq_len, n_features)``.

        Returns:
            1-D numpy array of predicted RUL values clipped to
            ``[0, rul_cap]``.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).squeeze(1).cpu().numpy()

        preds = np.clip(preds, 0.0, self._rul_cap)
        return preds.astype(np.float32)

    def predict_health_score(self, X: np.ndarray) -> np.ndarray:
        """Convert RUL predictions to Vehicle Health Scores (0–100).

        Uses the same linear mapping as Module 1:
        ``health_score = clip((RUL / rul_cap) * 100, 0, 100)``

        Args:
            X: Sequences array ``(n_windows, seq_len, n_features)``.

        Returns:
            1-D numpy array of health scores in ``[0, 100]``.
        """
        rul_preds = self.predict_rul(X)
        scores    = np.clip((rul_preds / self._rul_cap) * 100.0, 0.0, 100.0)
        return scores.astype(np.float32)

    def predict_fleet(
        self,
        X: np.ndarray,
        unit_ids: np.ndarray,
        last_cycles: np.ndarray,
    ) -> pd.DataFrame:
        """Generate a fleet-level prediction DataFrame.

        For each engine unit, only the prediction from its **last** sequence
        window (the most recent operational state) is retained.

        Args:
            X:           Sequences array ``(n_windows, seq_len, n_features)``.
            unit_ids:    Engine unit per window ``(n_windows,)``.
            last_cycles: Last observed cycle per window ``(n_windows,)``.

        Returns:
            DataFrame with one row per unique ``unit_id`` and columns::

                unit_id | predicted_rul | predicted_health_score | risk_tier | last_cycle

        Note:
            ``risk_tier`` uses the same thresholds as Module 1
            (from ``health_score.*_threshold`` in config).
        """
        rul_preds    = self.predict_rul(X)
        health_preds = self.predict_health_score(X)

        hs_cfg     = self._cfg["health_score"]
        healthy_t  = float(hs_cfg["healthy_threshold"])
        watch_t    = float(hs_cfg["watch_threshold"])
        warning_t  = float(hs_cfg["warning_threshold"])

        def _tier(score: float) -> str:
            if score >= healthy_t:
                return "HEALTHY"
            if score >= watch_t:
                return "WATCH"
            if score >= warning_t:
                return "WARNING"
            return "CRITICAL"

        rows: list[dict[str, Any]] = []
        for uid in np.sort(np.unique(unit_ids)):
            mask     = unit_ids == uid
            # Last window = the one with the highest index among this unit's windows
            last_idx = np.where(mask)[0][-1]
            rows.append(
                {
                    "unit_id":                int(uid),
                    "predicted_rul":          float(rul_preds[last_idx]),
                    "predicted_health_score": float(health_preds[last_idx]),
                    "risk_tier":              _tier(health_preds[last_idx]),
                    "last_cycle":             int(last_cycles[last_idx]),
                }
            )

        fleet_df = pd.DataFrame(rows)
        logger.info(
            "Fleet predictions complete  units=%d  mean_rul=%.1f  "
            "mean_health=%.1f",
            len(fleet_df),
            fleet_df["predicted_rul"].mean(),
            fleet_df["predicted_health_score"].mean(),
        )
        return fleet_df


# ── Standalone training entry-point ────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    # Ensure project root on sys.path when invoked as a module
    _ROOT = _Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from src.data_layer.loader import (
        compute_rul,
        drop_constant_sensors,
        load_raw_data,
        normalize,
        run_pipeline,
    )
    from src.utils.config_loader import load_config as _load_config

    cfg = _load_config()

    print("\n" + "=" * 60)
    print("  TRACE — Module 2: LSTM RUL Training")
    print("=" * 60)

    # ── 1. Load pipeline output from Module 1 ─────────────────────────────────
    pipeline = run_pipeline()
    X_train_raw = pipeline["X_train"]
    y_train     = pipeline["y_train"]
    X_test      = pipeline["X_test"]
    y_test      = pipeline["y_test"]

    # ── 2. Build sequences with unit_ids (needed for engine-level split) ───────
    df_train, df_test, df_rul = load_raw_data()
    df_train = compute_rul(df_train)
    df_train = drop_constant_sensors(df_train)

    window_size: int = cfg["data"]["window_size"]
    X_tr, y_tr, unit_ids_train = build_sequences_with_unit_ids(df_train, window_size)

    # Normalize using the already-fitted scaler saved by run_pipeline()
    scaler_path = cfg["data"]["scaler_path"]
    import joblib as _joblib
    scaler = _joblib.load(scaler_path)
    n_tr, ws, nf = X_tr.shape
    X_tr = scaler.transform(X_tr.reshape(-1, nf)).reshape(n_tr, ws, nf).astype("float32")

    # Also build test sequences with unit_ids for predict_fleet
    df_test_raw = df_test.copy()
    import numpy as _np
    df_rul_indexed = df_rul.reset_index(drop=True)
    df_rul_indexed["unit_id"] = df_rul_indexed.index + 1
    last_cycles_df = df_test_raw.groupby("unit_id")["cycle"].max().reset_index()
    last_cycles_df = last_cycles_df.merge(
        df_rul_indexed.rename(columns={"rul": "RUL"}), on="unit_id"
    )
    df_test_raw = df_test_raw.merge(
        last_cycles_df[["unit_id", "cycle", "RUL"]], on=["unit_id", "cycle"], how="left"
    )
    df_test_raw["RUL"] = df_test_raw.groupby("unit_id")["RUL"].transform(lambda s: s.bfill())
    for uid, grp_idx in df_test_raw.groupby("unit_id").groups.items():
        grp = df_test_raw.loc[grp_idx]
        if grp["RUL"].isna().any():
            last_rul   = grp["RUL"].dropna().iloc[-1]
            last_cycle = grp["cycle"].max()
            df_test_raw.loc[grp_idx, "RUL"] = last_rul + (last_cycle - grp["cycle"]).values
    df_test_raw["RUL"] = df_test_raw["RUL"].clip(upper=cfg["data"]["rul_cap"]).astype(_np.float32)
    df_test_clean = drop_constant_sensors(df_test_raw)

    X_te, y_te, unit_ids_test = build_sequences_with_unit_ids(df_test_clean, window_size)
    n_te, ws2, nf2 = X_te.shape
    X_te = scaler.transform(X_te.reshape(-1, nf2)).reshape(n_te, ws2, nf2).astype("float32")

    # last cycle per window (the cycle at the end of the window)
    last_cycles_per_window = _np.array([
        df_test_clean[df_test_clean["unit_id"] == uid]
            .sort_values("cycle")["cycle"].iloc[min(i + window_size - 1, len(
                df_test_clean[df_test_clean["unit_id"] == uid]) - 1)]
        for i, uid in enumerate(unit_ids_test)
    ], dtype=_np.int64)

    n_features = X_tr.shape[2]

    # ── 3. Train ───────────────────────────────────────────────────────────────
    trainer = Trainer(cfg, n_features=n_features)
    history = trainer.train(X_tr, y_tr, unit_ids_train)

    # Print first 5 and last 5 epoch losses
    print("\n--- Epoch Losses (first 5) ---")
    for i, (tl, vl) in enumerate(zip(history["train_losses"][:5], history["val_losses"][:5]), 1):
        print(f"  Epoch {i:03d}  train={tl:.4f}  val={vl:.4f}")
    if history["epochs_trained"] > 10:
        print("  ...")
        last5_start = max(5, history["epochs_trained"] - 5)
        for i, (tl, vl) in enumerate(
            zip(history["train_losses"][last5_start:], history["val_losses"][last5_start:]),
            start=last5_start + 1,
        ):
            print(f"  Epoch {i:03d}  train={tl:.4f}  val={vl:.4f}")

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    metrics = trainer.evaluate(X_te, y_te)
    metrics["epochs_trained"] = history["epochs_trained"]
    metrics["best_val_loss"]  = history["best_val_loss"]

    print(f"\n--- Test Set Metrics ---")
    print(f"  RMSE          : {metrics['rmse']:.4f}")
    print(f"  MAE           : {metrics['mae']:.4f}")
    print(f"  Epochs trained: {metrics['epochs_trained']}")

    # ── 5. Save ────────────────────────────────────────────────────────────────
    trainer.save(metrics)
    print(f"\n  ✓  Model saved  →  {cfg['model']['model_path']}")
    print(f"  ✓  Metadata     →  {cfg['model']['metadata_path']}")

    # ── 6. Fleet predictions ───────────────────────────────────────────────────
    predictor  = Predictor(cfg, n_features=n_features)
    fleet_preds = predictor.predict_fleet(X_te, unit_ids_test, last_cycles_per_window)

    print(f"\n--- Fleet Predictions (first 5 rows) ---")
    print(fleet_preds.head(5).to_string(index=False))
    print("\n" + "=" * 60)
