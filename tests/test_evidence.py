"""
Unit tests for the TRACE evidence layer (Module 4).

Covers event detection, ChromaDB logging, RAG retrieval, and idempotency.

Run with:
    pytest tests/test_evidence.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_timeline(
    unit_id: int = 1,
    n_cycles: int = 50,
    start_score: float = 100.0,
    drop_cycle: int | None = None,
    drop_amount: float = 10.0,
    cross_warning_at: int | None = None,
) -> pd.DataFrame:
    """Build a synthetic health timeline DataFrame for testing.

    Args:
        unit_id:          Engine unit ID.
        n_cycles:         Total number of cycles to generate.
        start_score:      Starting health score.
        drop_cycle:       Cycle index (0-based) at which to inject a sudden drop.
        drop_amount:      Size of the injected sudden drop.
        cross_warning_at: Cycle index at which to force score below 50 (Watch→Warning).

    Returns:
        DataFrame with columns: unit_id, cycle, health_score, risk_tier, s2…s9.
    """
    scores = np.linspace(start_score, max(start_score - 2 * n_cycles, 0), n_cycles)

    if drop_cycle is not None and drop_cycle < n_cycles:
        scores[drop_cycle:] -= drop_amount

    if cross_warning_at is not None:
        scores[cross_warning_at] = 19.0  # below warning threshold (20)
        scores[cross_warning_at + 1:] = np.linspace(
            19.0, 0.0, n_cycles - cross_warning_at - 1
        )

    # Clip to [0, 100]
    scores = np.clip(scores, 0, 100)

    def _tier(s: float) -> str:
        if s >= 80:
            return "HEALTHY"
        if s >= 50:
            return "WATCH"
        if s >= 20:
            return "WARNING"
        return "CRITICAL"

    rows = []
    for i, s in enumerate(scores):
        sensor_vals = {f"s{j}": float(50 + np.random.randn()) for j in range(2, 10)}
        row = {
            "unit_id": unit_id,
            "cycle": i + 1,
            "health_score": float(s),
            "risk_tier": _tier(s),
            "RUL": float(max(0, (100 - i) * 1.25)),
            **sensor_vals,
        }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def mock_config() -> dict[str, Any]:
    """Minimal config dict matching config.yaml schema."""
    return {
        "evidence": {
            "chromadb_persist_dir": "data/chromadb_test",
            "collection_name": "test_fleet_events",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k_results": 5,
            "health_drop_threshold": 5.0,
            "sustained_decline_window": 5,
            "recovery_threshold": 3.0,
            "relevance_display_threshold": 0.0,  # no filtering in tests
        }
    }


@pytest.fixture(scope="module")
def mock_embedder():
    """Fake embedder that returns deterministic 384-d vectors."""
    embedder = MagicMock()
    embedder.encode = MagicMock(
        side_effect=lambda texts, **_: np.random.default_rng(42).random(
            (len(texts), 384)
        ).astype("float32")
    )
    return embedder


@pytest.fixture(scope="module")
def in_memory_chroma():
    """Return an in-memory ChromaDB client for isolated tests."""
    try:
        import chromadb  # type: ignore
        return chromadb.EphemeralClient()
    except ImportError:
        pytest.skip("chromadb not installed")


@pytest.fixture(scope="module")
def event_logger(mock_config, in_memory_chroma, mock_embedder):
    """EventLogger wired to an in-memory ChromaDB + fake embedder."""
    from src.evidence_layer.event_logger import EventLogger

    el = EventLogger.__new__(EventLogger)
    el._cfg = mock_config
    el._ecfg = mock_config["evidence"]
    el._client = in_memory_chroma
    el._collection = in_memory_chroma.get_or_create_collection(
        name="test_fleet_events",
        metadata={"hnsw:space": "cosine"},
    )
    el._embedder = mock_embedder
    return el


@pytest.fixture(scope="module")
def rag_retriever(mock_config, in_memory_chroma, mock_embedder):
    """RAGRetriever wired to the same in-memory ChromaDB + fake embedder."""
    from src.evidence_layer.rag_retriever import RAGRetriever

    return RAGRetriever(mock_config, in_memory_chroma, embedder=mock_embedder)


# ── Tests: event detection ────────────────────────────────────────────────────


class TestEventDetection:
    """Validate event detection logic in EventLogger.generate_events()."""

    def test_health_drop_events_detected(self, event_logger) -> None:
        """A sudden drop > threshold must produce a health_drop event at that cycle."""
        DROP_CYCLE = 20  # 0-based index
        DROP_AMOUNT = 12.0  # > threshold of 5
        timeline = _make_timeline(
            unit_id=1, n_cycles=40,
            drop_cycle=DROP_CYCLE, drop_amount=DROP_AMOUNT,
        )
        events = event_logger.generate_events(timeline)
        drop_events = [e for e in events if e.event_type == "health_drop"]
        # The drop should appear at cycle DROP_CYCLE + 1 (1-indexed)
        assert any(e.cycle == DROP_CYCLE + 1 for e in drop_events), (
            f"Expected health_drop at cycle {DROP_CYCLE + 1}, "
            f"got cycles: {[e.cycle for e in drop_events]}"
        )

    def test_critical_entry_detected(self, event_logger) -> None:
        """First WARNING/CRITICAL cycle must produce exactly one critical_entry event."""
        CROSS_AT = 25  # 0-based
        timeline = _make_timeline(
            unit_id=2, n_cycles=50,
            cross_warning_at=CROSS_AT,
        )
        events = event_logger.generate_events(timeline)
        critical_events = [e for e in events if e.event_type == "critical_entry"]
        assert len(critical_events) == 1, (
            f"Expected exactly 1 critical_entry, got {len(critical_events)}"
        )
        assert critical_events[0].cycle == CROSS_AT + 1, (
            f"Expected critical_entry at cycle {CROSS_AT + 1}, "
            f"got {critical_events[0].cycle}"
        )

    def test_event_ids_unique(self, event_logger) -> None:
        """No two events across a multi-unit timeline may share an event_id."""
        timelines = [_make_timeline(uid, n_cycles=60) for uid in range(1, 6)]
        combined = pd.concat(timelines, ignore_index=True)
        events = event_logger.generate_events(combined)
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids)), (
            f"Duplicate event_ids found: "
            f"{[i for i in ids if ids.count(i) > 1][:5]}"
        )


# ── Tests: ChromaDB logging ───────────────────────────────────────────────────


class TestEventLogging:
    """Validate ChromaDB logging behaviour."""

    def test_chromadb_count_after_logging(
        self, event_logger, in_memory_chroma
    ) -> None:
        """After log_events(), the collection must contain exactly len(events) docs."""
        # Use a fresh collection to avoid pollution from other tests
        fresh_col = in_memory_chroma.get_or_create_collection(
            name="test_count_col",
            metadata={"hnsw:space": "cosine"},
        )
        original_col = event_logger._collection
        event_logger._collection = fresh_col

        try:
            timeline = _make_timeline(unit_id=10, n_cycles=50)
            events = event_logger.generate_events(timeline)
            assert len(events) > 0, "No events generated — adjust test timeline"
            event_logger.log_events(events)
            assert fresh_col.count() == len(events), (
                f"Expected {len(events)} events, got {fresh_col.count()}"
            )
        finally:
            event_logger._collection = original_col

    def test_one_time_setup_idempotent(self, event_logger, in_memory_chroma) -> None:
        """Calling setup() twice must not change the event count on the second call."""
        idempotent_col = in_memory_chroma.get_or_create_collection(
            name="test_idempotent_col",
            metadata={"hnsw:space": "cosine"},
        )
        original_col = event_logger._collection
        event_logger._collection = idempotent_col

        try:
            timeline = _make_timeline(unit_id=20, n_cycles=40)
            event_logger.setup(timeline)
            count_after_first = idempotent_col.count()
            assert count_after_first > 0

            # Second call — must skip logging
            event_logger.setup(timeline)
            count_after_second = idempotent_col.count()

            assert count_after_first == count_after_second, (
                f"setup() changed count from {count_after_first} to "
                f"{count_after_second} on second call"
            )
        finally:
            event_logger._collection = original_col


# ── Tests: RAG retrieval ──────────────────────────────────────────────────────


class TestRAGRetriever:
    """Validate retrieval filtering and vehicle scoping."""

    @pytest.fixture(autouse=True)
    def _seed_collection(self, event_logger, in_memory_chroma, mock_embedder):
        """Pre-populate a retrieval test collection with known events."""
        self._ret_col = in_memory_chroma.get_or_create_collection(
            name="test_retrieval_col",
            metadata={"hnsw:space": "cosine"},
        )
        # Insert two events for unit 5 and one for unit 7
        docs = [
            "Engine 5 experienced a health drop at cycle 10.",
            "Engine 5 entered WARNING status at cycle 20.",
            "Engine 7 shows sustained degradation.",
        ]
        metas = [
            {"unit_id": 5, "cycle": 10, "event_type": "health_drop",
             "health_score_after": 45.0, "risk_tier": "WARNING", "severity": 0.4},
            {"unit_id": 5, "cycle": 20, "event_type": "critical_entry",
             "health_score_after": 19.0, "risk_tier": "CRITICAL", "severity": 0.8},
            {"unit_id": 7, "cycle": 15, "event_type": "sustained_decline",
             "health_score_after": 60.0, "risk_tier": "WATCH", "severity": 0.3},
        ]
        ids = ["5_10_health_drop", "5_20_critical_entry", "7_15_sustained_decline"]
        embeddings = mock_embedder.encode(docs).tolist()
        self._ret_col.upsert(
            ids=ids, documents=docs, metadatas=metas, embeddings=embeddings
        )

        # Point retriever at this collection
        from src.evidence_layer.rag_retriever import RAGRetriever
        from src.utils.config_loader import load_config
        cfg = load_config()
        cfg["evidence"]["relevance_display_threshold"] = 0.0
        self._retriever = RAGRetriever(cfg, in_memory_chroma, embedder=mock_embedder)
        self._retriever._collection = self._ret_col
        self._retriever._relevance_threshold = 0.0

    def test_retrieve_filters_by_vehicle(self) -> None:
        """retrieve_for_vehicle(5) must return only events with unit_id == 5."""
        results = self._retriever.retrieve_for_vehicle(unit_id=5, top_k=10)
        assert len(results) > 0, "No events returned for unit 5"
        assert all(r["unit_id"] == 5 for r in results), (
            f"Got events for wrong units: {[r['unit_id'] for r in results]}"
        )

    def test_relevance_threshold_applied(self) -> None:
        """Events below the threshold must be filtered out."""
        # Set a very high threshold so all results are filtered
        original_threshold = self._retriever._relevance_threshold
        self._retriever._relevance_threshold = 999.0
        try:
            results = self._retriever.retrieve_for_vehicle(unit_id=5, top_k=5)
            assert len(results) == 0, (
                f"Expected 0 events above threshold=999, got {len(results)}"
            )
        finally:
            self._retriever._relevance_threshold = original_threshold
