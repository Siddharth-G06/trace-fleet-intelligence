"""
TRACE — Module 4: Evidence Layer — Event Logger

Converts raw health-score timelines into structured :class:`HealthEvent`
objects and persists them in a ChromaDB vector collection for later
semantic retrieval.

Four event types are detected per engine unit:
- ``health_drop``       — sudden score decrease > threshold in a single cycle
- ``critical_entry``   — first cycle where tier becomes WARNING or CRITICAL
- ``sustained_decline``— monotonically falling score over N consecutive cycles
- ``recovery``         — score increases > threshold after ≥5 declining cycles

All parameters come from ``config/config.yaml`` (``evidence.*``).
No ``print`` statements; all logging via :func:`~src.utils.logger.get_logger`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sensor columns present after dropping constant sensors
_SENSOR_COLS = [f"s{i}" for i in range(1, 22)
                if i not in (1, 5, 6, 10, 16, 18, 19)]


# ── HealthEvent dataclass ─────────────────────────────────────────────────────

@dataclass
class HealthEvent:
    """A single detected fleet health event.

    Attributes:
        event_id:           Unique identifier ``f"{unit_id}_{cycle}_{event_type}"``.
        unit_id:            Engine unit identifier.
        cycle:              Operational cycle at which the event occurred.
        event_type:         One of ``health_drop``, ``critical_entry``,
                            ``sustained_decline``, ``recovery``.
        health_score_before: Health score at the cycle *preceding* the event.
        health_score_after:  Health score at the event cycle.
        change_magnitude:   Absolute difference ``|after - before|``.
        risk_tier:          Risk tier label at the event cycle.
        sensor_context:     Top-3 sensors with highest deviation from unit
                            baseline, as ``{sensor_name: deviation_value}``.
        description:        Natural-language description used for embedding.
        severity:           Normalised severity in ``[0, 1]``.
    """

    event_id: str
    unit_id: int
    cycle: int
    event_type: str
    health_score_before: float
    health_score_after: float
    change_magnitude: float
    risk_tier: str
    sensor_context: dict[str, float]
    description: str
    severity: float


# ── EventLogger ───────────────────────────────────────────────────────────────

class EventLogger:
    """Detect fleet health events from a per-cycle timeline and log to ChromaDB.

    Args:
        config:        Full parsed configuration dict from ``config.yaml``.
        chroma_client: Pre-initialised ``chromadb.Client`` or
                       ``chromadb.PersistentClient`` instance.

    Example::

        import chromadb
        from src.evidence_layer.event_logger import EventLogger
        from src.utils.config_loader import load_config

        cfg = load_config()
        client = chromadb.PersistentClient(path=cfg["evidence"]["chromadb_persist_dir"])
        el = EventLogger(cfg, client)
        el.setup(health_timeline_df)
        print(el.get_event_stats())
    """

    def __init__(self, config: dict[str, Any], chroma_client: Any) -> None:
        self._cfg = config
        self._ecfg: dict[str, Any] = config.get("evidence", {})
        self._client = chroma_client

        collection_name: str = self._ecfg.get(
            "collection_name", "trace_fleet_events"
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Load sentence-transformer once
        model_name: str = self._ecfg.get("embedding_model", "all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._embedder = SentenceTransformer(model_name)

        logger.info(
            "EventLogger ready  collection=%s  model=%s  existing_events=%d",
            collection_name,
            model_name,
            self._collection.count(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_events(self, health_timeline: pd.DataFrame) -> list[HealthEvent]:
        """Detect all health events across every engine unit in *health_timeline*.

        Args:
            health_timeline: Per-cycle DataFrame with columns ``unit_id``,
                ``cycle``, ``health_score``, ``risk_tier``, and sensor columns
                ``s2``…``s21`` (after constant-sensor dropping).

        Returns:
            Flat list of :class:`HealthEvent` objects, ordered by
            ``(unit_id, cycle)``.
        """
        drop_thresh: float = float(
            self._ecfg.get("health_drop_threshold", 5.0)
        )
        decline_window: int = int(
            self._ecfg.get("sustained_decline_window", 10)
        )
        recovery_thresh: float = float(
            self._ecfg.get("recovery_threshold", 3.0)
        )

        # Pre-compute per-unit sensor baselines (mean of each sensor)
        sensor_cols = [c for c in _SENSOR_COLS if c in health_timeline.columns]
        unit_baselines: dict[int, pd.Series] = {}
        for uid, grp in health_timeline.groupby("unit_id"):
            unit_baselines[int(uid)] = grp[sensor_cols].mean()

        all_events: list[HealthEvent] = []
        critical_tiers = {"WARNING", "CRITICAL"}

        for uid, group in health_timeline.groupby("unit_id"):
            uid_int = int(uid)
            grp = group.sort_values("cycle").reset_index(drop=True)
            scores = grp["health_score"].values
            cycles = grp["cycle"].values
            tiers = grp["risk_tier"].values
            baseline = unit_baselines.get(uid_int, pd.Series(dtype=float))

            entered_critical = False

            for i in range(1, len(grp)):
                score_prev = float(scores[i - 1])
                score_curr = float(scores[i])
                cycle_curr = int(cycles[i])
                tier_curr = str(tiers[i])
                row = grp.iloc[i]

                # ── 1. Health drop ────────────────────────────────────────────
                drop = score_prev - score_curr
                if drop > drop_thresh:
                    sensor_ctx = self._sensor_context(row, baseline, sensor_cols)
                    severity = min(drop / 20.0, 1.0)
                    ev = HealthEvent(
                        event_id=f"{uid_int}_{cycle_curr}_health_drop",
                        unit_id=uid_int,
                        cycle=cycle_curr,
                        event_type="health_drop",
                        health_score_before=score_prev,
                        health_score_after=score_curr,
                        change_magnitude=drop,
                        risk_tier=tier_curr,
                        sensor_context=sensor_ctx,
                        description="",
                        severity=severity,
                    )
                    ev.description = self._generate_description(ev)
                    all_events.append(ev)

                # ── 2. Critical entry ─────────────────────────────────────────
                if not entered_critical and tier_curr in critical_tiers:
                    entered_critical = True
                    peak_score = float(scores[:i].max())
                    magnitude = peak_score - score_curr
                    severity = 1.0 - score_curr / 100.0
                    sensor_ctx = self._sensor_context(row, baseline, sensor_cols)
                    ev = HealthEvent(
                        event_id=f"{uid_int}_{cycle_curr}_critical_entry",
                        unit_id=uid_int,
                        cycle=cycle_curr,
                        event_type="critical_entry",
                        health_score_before=peak_score,
                        health_score_after=score_curr,
                        change_magnitude=magnitude,
                        risk_tier=tier_curr,
                        sensor_context=sensor_ctx,
                        description="",
                        severity=severity,
                    )
                    ev.description = self._generate_description(ev)
                    all_events.append(ev)

            # ── 3. Sustained decline ──────────────────────────────────────────
            for start in range(len(grp) - decline_window):
                window_scores = scores[start: start + decline_window]
                if all(
                    window_scores[j] > window_scores[j + 1]
                    for j in range(len(window_scores) - 1)
                ):
                    end_idx = start + decline_window - 1
                    score_first = float(window_scores[0])
                    score_last = float(window_scores[-1])
                    cycle_end = int(cycles[end_idx])
                    row_end = grp.iloc[end_idx]
                    rate = (score_first - score_last) / decline_window
                    sensor_ctx = self._sensor_context(
                        row_end, baseline, sensor_cols
                    )
                    severity = min((score_first - score_last) / 20.0, 1.0)
                    ev_id = f"{uid_int}_{cycle_end}_sustained_decline"
                    # Deduplicate — only emit one event per ending cycle
                    if not any(e.event_id == ev_id for e in all_events):
                        ev = HealthEvent(
                            event_id=ev_id,
                            unit_id=uid_int,
                            cycle=cycle_end,
                            event_type="sustained_decline",
                            health_score_before=score_first,
                            health_score_after=score_last,
                            change_magnitude=score_first - score_last,
                            risk_tier=str(tiers[end_idx]),
                            sensor_context=sensor_ctx,
                            description="",
                            severity=severity,
                        )
                        ev._window = decline_window  # type: ignore[attr-defined]
                        ev._rate = rate              # type: ignore[attr-defined]
                        ev.description = self._generate_description(ev)
                        all_events.append(ev)

            # ── 4. Recovery ───────────────────────────────────────────────────
            declining_streak = 0
            for i in range(1, len(grp)):
                score_prev = float(scores[i - 1])
                score_curr = float(scores[i])
                cycle_curr = int(cycles[i])
                tier_curr = str(tiers[i])
                row = grp.iloc[i]

                if score_curr < score_prev:
                    declining_streak += 1
                elif (
                    score_curr - score_prev > recovery_thresh
                    and declining_streak >= 5
                ):
                    recovery_amount = score_curr - score_prev
                    sensor_ctx = self._sensor_context(row, baseline, sensor_cols)
                    severity = min(recovery_amount / 20.0, 1.0)
                    ev_id = f"{uid_int}_{cycle_curr}_recovery"
                    if not any(e.event_id == ev_id for e in all_events):
                        ev = HealthEvent(
                            event_id=ev_id,
                            unit_id=uid_int,
                            cycle=cycle_curr,
                            event_type="recovery",
                            health_score_before=score_prev,
                            health_score_after=score_curr,
                            change_magnitude=recovery_amount,
                            risk_tier=tier_curr,
                            sensor_context=sensor_ctx,
                            description="",
                            severity=severity,
                        )
                        ev.description = self._generate_description(ev)
                        all_events.append(ev)
                    declining_streak = 0
                else:
                    declining_streak = 0

        logger.info(
            "Events generated  total=%d  by_type=%s",
            len(all_events),
            {
                et: sum(1 for e in all_events if e.event_type == et)
                for et in (
                    "health_drop", "critical_entry",
                    "sustained_decline", "recovery",
                )
            },
        )
        return all_events

    def log_events(self, events: list[HealthEvent]) -> int:
        """Embed and upsert events into ChromaDB in batches of 100.

        Args:
            events: List of :class:`HealthEvent` objects to persist.

        Returns:
            Total number of events successfully logged.
        """
        if not events:
            logger.warning("log_events called with empty events list")
            return 0

        batch_size = 100
        logged = 0

        for batch_start in range(0, len(events), batch_size):
            batch = events[batch_start: batch_start + batch_size]
            docs = [e.description for e in batch]
            ids = [e.event_id for e in batch]

            # Flatten to scalar metadata (ChromaDB cannot store nested dicts)
            metas = []
            for e in batch:
                metas.append({
                    "unit_id":              e.unit_id,
                    "cycle":                e.cycle,
                    "event_type":           e.event_type,
                    "health_score_before":  round(e.health_score_before, 2),
                    "health_score_after":   round(e.health_score_after, 2),
                    "change_magnitude":     round(e.change_magnitude, 2),
                    "risk_tier":            e.risk_tier,
                    "severity":             round(e.severity, 4),
                    "sensor_context_json":  json.dumps(
                        {k: round(v, 4) for k, v in e.sensor_context.items()}
                    ),
                })

            # Compute embeddings for this batch
            embeddings = self._embedder.encode(docs, show_progress_bar=False).tolist()

            self._collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
            logged += len(batch)

        # Log per-type breakdown
        by_type: dict[str, int] = {}
        for e in events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
        logger.info("Events logged to ChromaDB  total=%d  by_type=%s", logged, by_type)
        return logged

    def setup(self, health_timeline: pd.DataFrame) -> None:
        """Idempotent first-run setup: generate and log events only when empty.

        Checks ``collection.count()`` — if 0, runs the full pipeline;
        otherwise logs a "already loaded" message and skips re-indexing.

        Args:
            health_timeline: Per-cycle health DataFrame (same schema as
                :meth:`generate_events`).
        """
        existing = self._collection.count()
        if existing == 0:
            logger.info("First run — generating and logging fleet events…")
            events = self.generate_events(health_timeline)
            n = self.log_events(events)
            logger.info("First run complete — logged %d events", n)
        else:
            logger.info(
                "Loaded existing %d events from ChromaDB — skipping re-index",
                existing,
            )

    def get_event_stats(self) -> dict[str, Any]:
        """Return summary statistics for the persisted event collection.

        Returns:
            Dict with keys:
            - ``total_events`` (int)
            - ``events_per_type`` (dict[str, int])
            - ``engines_with_events`` (int)
        """
        total = self._collection.count()
        if total == 0:
            return {
                "total_events": 0,
                "events_per_type": {},
                "engines_with_events": 0,
            }

        # Fetch all metadata (no documents needed)
        result = self._collection.get(include=["metadatas"])
        metas = result.get("metadatas", []) or []

        by_type: dict[str, int] = {}
        unit_ids: set[int] = set()
        for m in metas:
            et = str(m.get("event_type", "unknown"))
            by_type[et] = by_type.get(et, 0) + 1
            uid = m.get("unit_id")
            if uid is not None:
                unit_ids.add(int(uid))

        return {
            "total_events":      total,
            "events_per_type":   by_type,
            "engines_with_events": len(unit_ids),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sensor_context(
        self,
        row: pd.Series,
        baseline: pd.Series,
        sensor_cols: list[str],
    ) -> dict[str, float]:
        """Return the top-3 sensors with highest deviation from unit baseline.

        Args:
            row:         A single DataFrame row with sensor readings.
            baseline:    Per-sensor mean values for this unit.
            sensor_cols: List of sensor column names present in *row*.

        Returns:
            Dict of ``{sensor_name: abs_deviation}`` for the top 3 sensors.
        """
        if baseline.empty or not sensor_cols:
            return {}

        deviations: dict[str, float] = {}
        for col in sensor_cols:
            try:
                val = float(row[col])
                base = float(baseline[col])
                deviations[col] = abs(val - base)
            except (KeyError, TypeError, ValueError):
                pass

        top3 = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:3]
        return {k: round(v, 4) for k, v in top3}

    def _generate_description(self, event: HealthEvent) -> str:
        """Generate a natural-language description used for vector embedding.

        Args:
            event: Partially-populated :class:`HealthEvent` (``description``
                   field may be empty at this point).

        Returns:
            Human-readable description string.
        """
        uid = event.unit_id
        cycle = event.cycle
        before = event.health_score_before
        after = event.health_score_after
        mag = event.change_magnitude
        tier = event.risk_tier
        ctx = event.sensor_context

        if event.event_type == "health_drop":
            return (
                f"Engine {uid} experienced a sudden health drop at cycle {cycle}. "
                f"Score fell {mag:.1f} points from {before:.1f} to {after:.1f}. "
                f"Risk tier: {tier}. "
                f"Primary sensor deviations: {ctx}."
            )

        if event.event_type == "critical_entry":
            return (
                f"Engine {uid} entered {tier} status at cycle {cycle} with health "
                f"score {after:.1f}. This marks the transition from stable operation "
                f"to active degradation. "
                f"Accumulated decline since peak: {mag:.1f} points."
            )

        if event.event_type == "sustained_decline":
            window = getattr(event, "_window", 10)
            rate = getattr(event, "_rate", mag / max(window, 1))
            return (
                f"Engine {uid} shows sustained degradation across {window} consecutive "
                f"cycles ending at cycle {cycle}. "
                f"Health declined from {before:.1f} to {after:.1f} without recovery. "
                f"Decline rate: {rate:.2f} points per cycle."
            )

        if event.event_type == "recovery":
            return (
                f"Engine {uid} showed health improvement at cycle {cycle}. "
                f"Score recovered {mag:.1f} points from {before:.1f} to {after:.1f}. "
                f"Brief recovery in otherwise declining trajectory."
            )

        return (
            f"Engine {uid} health event at cycle {cycle}: "
            f"{before:.1f} → {after:.1f} (Δ{mag:.1f}). Tier: {tier}."
        )


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import chromadb  # type: ignore

    from src.data_layer.loader import compute_rul, load_raw_data
    from src.data_layer.health_score import compute_health_scores
    from src.utils.config_loader import load_config

    cfg = load_config()
    rul_cap: int = cfg["data"]["rul_cap"]
    ecfg = cfg["evidence"]

    # Build health timeline
    df_train, _, _ = load_raw_data()
    df_train = compute_rul(df_train)
    df_scored = compute_health_scores(df_train, rul_cap)

    # Init ChromaDB
    Path(ecfg["chromadb_persist_dir"]).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=ecfg["chromadb_persist_dir"])

    el = EventLogger(cfg, client)
    el.setup(df_scored)

    stats = el.get_event_stats()
    print("\n── Event Stats ──────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Sample 3 descriptions
    if stats["total_events"] > 0:
        sample = client.get_or_create_collection(ecfg["collection_name"]).get(
            limit=3, include=["documents", "metadatas"]
        )
        print("\n── Sample Event Descriptions ────────────────────")
        for i, doc in enumerate(sample.get("documents", []), 1):
            meta = sample["metadatas"][i - 1]
            print(f"\n  [{i}] {meta.get('event_type')} | Engine {meta.get('unit_id')} | Cycle {meta.get('cycle')}")
            print(f"      {doc}")
