"""
Data loading and preprocessing pipeline for the TRACE fleet intelligence system.

Handles raw CMAPSS FD001 data ingestion, RUL computation, sensor dropping,
sequence building, and feature normalization.  All parameters are read from
``config/config.yaml``; no magic numbers are hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Column schema ──────────────────────────────────────────────────────────────
_SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMN_NAMES: list[str] = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + _SENSOR_COLS
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not already exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw FD001 train, test, and RUL text files.

    File paths are resolved from ``config.yaml``.  The function reads
    space-delimited text files (no header, trailing whitespace stripped) and
    assigns the canonical :data:`COLUMN_NAMES` schema.

    Returns:
        A three-tuple ``(df_train, df_test, df_rul)`` where:

        * ``df_train`` — training engine runs (all cycles to failure)
        * ``df_test``  — truncated test engine runs
        * ``df_rul``   — one-row-per-unit ground-truth RUL for the test set

    Raises:
        FileNotFoundError: If any of the configured data paths are missing.
    """
    cfg = load_config()
    data_cfg = cfg["data"]

    def _read(path: str, cols: list[str]) -> pd.DataFrame:
        full_path = Path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: '{full_path.resolve()}'")
        df = pd.read_csv(
            full_path,
            sep=r"\s+",
            header=None,
            names=cols,
            engine="python",
        )
        # Strip any trailing NaN columns produced by trailing whitespace
        df = df.dropna(axis=1, how="all")
        logger.info("Loaded '%s'  shape=%s", full_path.name, df.shape)
        return df

    df_train = _read(data_cfg["train_path"], COLUMN_NAMES)
    df_test = _read(data_cfg["test_path"], COLUMN_NAMES)
    df_rul = _read(data_cfg["rul_path"], ["rul"])

    return df_train, df_test, df_rul


def compute_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    """Add a ``RUL`` column to the training DataFrame.

    For each engine unit the Remaining Useful Life at cycle *c* is:

    .. code-block:: text

        RUL = max_cycle_for_unit - c

    Values are then clipped at ``rul_cap`` from config so the model is not
    penalised for predicting degradation early when the engine is still
    nominally healthy.

    Args:
        df_train: Raw training DataFrame produced by :func:`load_raw_data`.

    Returns:
        A copy of *df_train* with a new integer ``RUL`` column appended.
    """
    cfg = load_config()
    rul_cap: int = cfg["data"]["rul_cap"]

    df = df_train.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    df.drop(columns=["max_cycle"], inplace=True)

    logger.info(
        "RUL computed — cap=%d  RUL range=[%d, %d]",
        rul_cap,
        df["RUL"].min(),
        df["RUL"].max(),
    )
    return df


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensor columns that carry no degradation signal.

    The list of sensors to drop is read from ``config.yaml``
    (``data.sensors_to_drop``).  Only columns that actually exist in *df*
    are dropped, so the function is safe to call on both the train and test
    DataFrames.

    Args:
        df: DataFrame containing sensor columns.

    Returns:
        DataFrame with the configured low-variance sensors removed.
    """
    cfg = load_config()
    to_drop: list[str] = cfg["data"]["sensors_to_drop"]

    existing_drops = [c for c in to_drop if c in df.columns]
    df_out = df.drop(columns=existing_drops)
    logger.info("Dropped sensors %s  remaining columns=%d", existing_drops, len(df_out.columns))
    return df_out


def build_sequences(
    df: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for sequence-to-one RUL prediction.

    For each engine unit, a window of *window_size* consecutive cycles is
    extracted.  The feature matrix for each window is the sensor readings
    over those cycles, and the label is the RUL at the **last** cycle of
    the window.

    Args:
        df: DataFrame with sensor columns and a ``RUL`` column.
            Must **not** contain ``unit_id`` or ``cycle`` in the feature set
            (they are excluded internally).
        window_size: Number of cycles per sequence window.

    Returns:
        A tuple ``(X, y)`` where:

        * ``X`` has shape ``(n_sequences, window_size, n_features)``
        * ``y`` has shape ``(n_sequences,)``
    """
    feature_cols = [
        c for c in df.columns if c not in ("unit_id", "cycle", "RUL")
    ]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle")
        features = group[feature_cols].values  # (n_cycles, n_features)
        labels = group["RUL"].values           # (n_cycles,)

        n_windows = len(features) - window_size + 1
        for i in range(n_windows):
            X_list.append(features[i : i + window_size])
            y_list.append(labels[i + window_size - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(
        "Sequences built  X.shape=%s  y.shape=%s  window_size=%d",
        X.shape,
        y.shape,
        window_size,
    )
    return X, y


def normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit a MinMaxScaler on train data and transform both train and test.

    The scaler is fitted on a 2-D reshape of *X_train* so every feature
    across all time-steps is scaled to ``[0, 1]``.  The fitted scaler is
    persisted to disk (path from config) for later use during inference.

    Args:
        X_train: Training sequences array of shape
            ``(n_train, window_size, n_features)``.
        X_test: Test sequences array of shape
            ``(n_test, window_size, n_features)``.

    Returns:
        A three-tuple ``(X_train_scaled, X_test_scaled, scaler)`` where
        both scaled arrays have the same shape as their inputs.
    """
    cfg = load_config()
    scaler_path: str = cfg["data"]["scaler_path"]
    _ensure_dir(scaler_path)

    n_train, window_size, n_features = X_train.shape

    scaler = MinMaxScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape).astype(np.float32)

    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to '%s'", scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def run_pipeline() -> dict[str, Any]:
    """Execute the full data preparation pipeline end-to-end.

    Steps performed:

    1. Load raw FD001 train/test/RUL files.
    2. Compute RUL labels (train) and merge ground-truth RUL (test).
    3. Drop low-variance sensor columns.
    4. Build sliding-window sequences.
    5. Normalize features with a MinMaxScaler fitted on train data only.

    Returns:
        A dict with the following keys:

        * ``X_train``      — ``(n_train, window_size, n_features)``
        * ``y_train``      — ``(n_train,)``
        * ``X_test``       — ``(n_test, window_size, n_features)``
        * ``y_test``       — ``(n_test,)``  (ground-truth RUL per test unit)
        * ``feature_names``— list of sensor column names after dropping
        * ``scaler``       — fitted :class:`~sklearn.preprocessing.MinMaxScaler`
    """
    cfg = load_config()
    data_cfg = cfg["data"]
    window_size: int = data_cfg["window_size"]

    logger.info("=== TRACE Data Pipeline START ===")

    # ── 1. Load ────────────────────────────────────────────────────────────────
    df_train, df_test, df_rul = load_raw_data()

    # ── 2. RUL labels ──────────────────────────────────────────────────────────
    df_train = compute_rul(df_train)

    # For the test set, the ground-truth RUL for each unit is the value in
    # RUL_FD001.txt.  We assign it to the LAST cycle of each test unit.
    df_test = df_test.copy()
    df_rul = df_rul.reset_index(drop=True)
    df_rul["unit_id"] = df_rul.index + 1  # unit IDs are 1-indexed

    # Get the last cycle per test unit and attach the ground-truth RUL
    last_cycles = df_test.groupby("unit_id")["cycle"].max().reset_index()
    last_cycles = last_cycles.merge(df_rul.rename(columns={"rul": "RUL"}), on="unit_id")

    df_test = df_test.merge(
        last_cycles[["unit_id", "cycle", "RUL"]], on=["unit_id", "cycle"], how="left"
    )
    # Back-fill RUL for earlier cycles in each unit (RUL increases going back)
    df_test["RUL"] = df_test.groupby("unit_id")["RUL"].transform(lambda s: s.bfill())

    # Where still NaN (cycles before the last), reconstruct RUL = last_rul + (last_cycle - cycle)
    # Use explicit per-unit index update to avoid pandas groupby.apply deprecation warnings.
    for uid, grp_idx in df_test.groupby("unit_id").groups.items():
        grp = df_test.loc[grp_idx]
        if grp["RUL"].isna().any():
            last_rul = grp["RUL"].dropna().iloc[-1]
            last_cycle = grp["cycle"].max()
            df_test.loc[grp_idx, "RUL"] = last_rul + (last_cycle - grp["cycle"]).values

    df_test["RUL"] = df_test["RUL"].clip(upper=data_cfg["rul_cap"]).astype(np.float32)

    # ── 3. Drop constant sensors ───────────────────────────────────────────────
    df_train_clean = drop_constant_sensors(df_train)
    df_test_clean = drop_constant_sensors(df_test)

    # ── 4. Feature names ───────────────────────────────────────────────────────
    feature_names: list[str] = [
        c for c in df_train_clean.columns if c not in ("unit_id", "cycle", "RUL")
    ]
    logger.info("Feature names (%d): %s", len(feature_names), feature_names)

    # ── 5. Build sequences ─────────────────────────────────────────────────────
    X_train, y_train = build_sequences(df_train_clean, window_size)
    X_test, y_test = build_sequences(df_test_clean, window_size)

    # ── 6. Normalize ───────────────────────────────────────────────────────────
    X_train, X_test, scaler = normalize(X_train, X_test)

    logger.info("=== TRACE Data Pipeline COMPLETE ===")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
    }


# ── Standalone runner ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_pipeline()
    print("\n--- Pipeline Output Shapes ---")
    for key, value in result.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value).__name__}")
