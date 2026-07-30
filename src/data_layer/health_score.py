"""
Fleet health scoring for the TRACE fleet intelligence system.

Converts raw RUL values into a normalized 0–100 health score and assigns
each engine unit to a risk tier based on configurable thresholds.
"""

from __future__ import annotations

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Risk tier labels (ordered worst → best for clarity) ───────────────────────
CRITICAL = "CRITICAL"
WARNING = "WARNING"
WATCH = "WATCH"
HEALTHY = "HEALTHY"


def compute_health_scores(
    df_train: pd.DataFrame,
    rul_cap: int,
) -> pd.DataFrame:
    """Add ``health_score`` and ``risk_tier`` columns to the training DataFrame.

    Health score is a linear mapping of RUL to the ``[0, 100]`` range:

    .. code-block:: text

        health_score = clip((RUL / rul_cap) * 100,  0, 100)

    Risk tiers are assigned based on thresholds read from
    ``config.yaml`` (``health_score.*_threshold``):

    * ``HEALTHY``  — score ≥ healthy_threshold
    * ``WATCH``    — watch_threshold ≤ score < healthy_threshold
    * ``WARNING``  — warning_threshold ≤ score < watch_threshold
    * ``CRITICAL`` — score < warning_threshold

    Args:
        df_train: Training DataFrame that already contains a ``RUL`` column
            (as produced by :func:`~src.data_layer.loader.compute_rul`).
        rul_cap: Maximum RUL value used as the 100 % health reference.

    Returns:
        A copy of *df_train* with two additional columns:
        ``health_score`` (float) and ``risk_tier`` (str).
    """
    cfg = load_config()
    hs_cfg = cfg["health_score"]
    healthy_t: float = float(hs_cfg["healthy_threshold"])
    watch_t: float = float(hs_cfg["watch_threshold"])
    warning_t: float = float(hs_cfg["warning_threshold"])

    df = df_train.copy()

    # ── Health score ──────────────────────────────────────────────────────────
    df["health_score"] = ((df["RUL"] / rul_cap) * 100).clip(0.0, 100.0)

    # ── Risk tier ─────────────────────────────────────────────────────────────
    def _assign_tier(score: float) -> str:
        if score >= healthy_t:
            return HEALTHY
        if score >= watch_t:
            return WATCH
        if score >= warning_t:
            return WARNING
        return CRITICAL

    df["risk_tier"] = df["health_score"].apply(_assign_tier)

    logger.info(
        "Health scores computed  rows=%d  tier_counts=%s",
        len(df),
        df["risk_tier"].value_counts().to_dict(),
    )
    return df


def get_fleet_snapshot(
    df_with_scores: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simulate a live fleet snapshot — one row per engine unit.

    In real-world fleet monitoring, vehicles are at **different points** in
    their operational lifecycle.  Picking the last training cycle for every
    unit would show RUL=0 for all (training data runs engines to failure),
    producing a misleading all-CRITICAL dashboard.

    This function simulates a realistic live snapshot by randomly sampling
    each unit at a different fraction of its total operational life
    (between 20 % and 85 %), so the fleet naturally spreads across all
    risk tiers (HEALTHY / WATCH / WARNING / CRITICAL).

    Args:
        df_with_scores: DataFrame produced by :func:`compute_health_scores`
            containing ``unit_id``, ``cycle``, ``RUL``, ``health_score``,
            and ``risk_tier`` columns.
        random_seed: Seed for reproducible sampling.  Change to generate a
            different but consistent fleet snapshot.

    Returns:
        DataFrame with one row per ``unit_id``, columns:

        * ``unit_id``        — engine identifier
        * ``current_cycle``  — the sampled cycle for this snapshot
        * ``health_score``   — health score (0–100) at the sampled cycle
        * ``risk_tier``      — risk classification string
        * ``rul_remaining``  — RUL at the sampled cycle

        Rows are sorted by ``health_score`` **ascending** so the most
        critical units appear at the top of the table.
    """
    import numpy as np

    rng = np.random.default_rng(random_seed)

    rows: list[dict] = []
    for unit_id, group in df_with_scores.groupby("unit_id"):
        group_sorted = group.sort_values("cycle")
        n = len(group_sorted)

        # Sample between 20 % and 85 % of the unit's life to get variety
        lo = max(0, int(n * 0.20))
        hi = max(lo + 1, int(n * 0.85))
        idx = rng.integers(lo, hi)

        row = group_sorted.iloc[idx]
        rows.append(
            {
                "unit_id": int(unit_id),
                "current_cycle": int(row["cycle"]),
                "health_score": float(row["health_score"]),
                "risk_tier": str(row["risk_tier"]),
                "rul_remaining": float(row["RUL"]),
            }
        )

    snapshot = pd.DataFrame(rows).sort_values(
        "health_score", ascending=True
    ).reset_index(drop=True)

    logger.info(
        "Fleet snapshot ready  units=%d  critical=%d  warning=%d  watch=%d  healthy=%d",
        len(snapshot),
        (snapshot["risk_tier"] == CRITICAL).sum(),
        (snapshot["risk_tier"] == WARNING).sum(),
        (snapshot["risk_tier"] == WATCH).sum(),
        (snapshot["risk_tier"] == HEALTHY).sum(),
    )
    return snapshot

# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_layer.loader import compute_rul, load_raw_data
    from src.utils.config_loader import load_config as _cfg

    cfg = _cfg()
    rul_cap: int = cfg["data"]["rul_cap"]

    df_train, _, _ = load_raw_data()
    df_train = compute_rul(df_train)
    df_scored = compute_health_scores(df_train, rul_cap)
    snapshot = get_fleet_snapshot(df_scored)

    print(f"\nFleet snapshot ({len(snapshot)} units):")
    print(snapshot.to_string(index=False))
