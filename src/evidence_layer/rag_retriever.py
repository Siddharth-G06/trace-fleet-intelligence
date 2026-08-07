"""
TRACE — Module 4: Evidence Layer — RAG Retriever

Retrieves the most semantically relevant :class:`~src.evidence_layer.event_logger.HealthEvent`
records from ChromaDB to support hypothesis verdicts produced by
:class:`~src.reasoning_layer.hypothesis_reasoner.HypothesisReasoner`.

The retriever is fully decoupled from :class:`~src.evidence_layer.event_logger.EventLogger`;
it only reads from the shared ChromaDB collection and never writes to it.

All parameters come from ``config/config.yaml`` (``evidence.*``).
No ``print`` statements; all logging via :func:`~src.utils.logger.get_logger`.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.reasoning_layer.hypothesis_reasoner import VerdictResult

logger = get_logger(__name__)


# ── Query templates keyed by query_type ──────────────────────────────────────

_QUERY_TEMPLATES: dict[str, str] = {
    "threshold_query": (
        "urgent maintenance health deterioration critical remaining useful life "
        "engines {vehicle_ids}"
    ),
    "ranking_query": (
        "most severe health degradation lowest health scores at-risk engines "
        "{vehicle_ids}"
    ),
    "anomaly_hunt": (
        "unusual health patterns anomalous behavior unexpected degradation "
        "{vehicle_ids}"
    ),
    "fleet_comparison": (
        "health decline rate degradation comparison group difference {vehicle_ids}"
    ),
    "trend_analysis": (
        "fleet health trend declining improving overall degradation pattern"
    ),
}


# ── RAGRetriever ──────────────────────────────────────────────────────────────

class RAGRetriever:
    """Semantic event retriever backed by ChromaDB + sentence-transformers.

    Accepts a pre-initialised ChromaDB client and a pre-loaded
    :class:`~sentence_transformers.SentenceTransformer` embedding model to
    avoid redundant model loading when used alongside
    :class:`~src.evidence_layer.event_logger.EventLogger`.

    Args:
        config:        Full parsed configuration dict from ``config.yaml``.
        chroma_client: Pre-initialised ``chromadb.PersistentClient``.
        embedder:      Optional pre-loaded ``SentenceTransformer`` instance.
                       If *None*, the model is loaded from
                       ``config.evidence.embedding_model``.

    Example::

        import chromadb
        from src.evidence_layer.rag_retriever import RAGRetriever
        from src.utils.config_loader import load_config

        cfg = load_config()
        client = chromadb.PersistentClient(path=cfg["evidence"]["chromadb_persist_dir"])
        retriever = RAGRetriever(cfg, client)
        events = retriever.retrieve_for_vehicle(unit_id=12)
        print(retriever.format_evidence_display(events))
    """

    def __init__(
        self,
        config: dict[str, Any],
        chroma_client: Any,
        embedder: Any = None,
    ) -> None:
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

        # Reuse caller-supplied embedder or load our own
        if embedder is not None:
            self._embedder = embedder
        else:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model_name: str = self._ecfg.get("embedding_model", "all-MiniLM-L6-v2")
            self._embedder = SentenceTransformer(model_name)
            logger.info("RAGRetriever loaded embedding model  model=%s", model_name)

        self._top_k: int = int(self._ecfg.get("top_k_results", 5))
        self._relevance_threshold: float = float(
            self._ecfg.get("relevance_display_threshold", 0.5)
        )

        logger.info(
            "RAGRetriever ready  collection=%s  events=%d  top_k=%d  "
            "relevance_threshold=%.2f",
            collection_name,
            self._collection.count(),
            self._top_k,
            self._relevance_threshold,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, verdict: "VerdictResult") -> list[dict[str, Any]]:
        """Retrieve the most relevant events for a given verdict.

        Builds a targeted query string from the verdict's ``query_type`` and
        ``supporting_vehicles``, then applies a ``unit_id`` ``$in`` filter
        when specific vehicles are cited.

        Args:
            verdict: A :class:`~src.reasoning_layer.hypothesis_reasoner.VerdictResult`
                produced by the hypothesis reasoner.

        Returns:
            List of event dicts sorted by ``relevance_score`` descending, with
            events below ``relevance_display_threshold`` removed.
        """
        query_type: str = verdict.query_type
        supporting: list[str] = verdict.supporting_vehicles

        # Extract numeric unit IDs from strings like "Engine 12"
        supporting_ids: list[int] = []
        for v in supporting:
            try:
                supporting_ids.append(int(v.replace("Engine ", "").strip()))
            except ValueError:
                pass

        # Build query string
        vehicle_ids_str = (
            " ".join(str(i) for i in supporting_ids)
            if supporting_ids
            else "fleet all vehicles"
        )
        template = _QUERY_TEMPLATES.get(
            query_type, _QUERY_TEMPLATES["ranking_query"]
        )
        query_string = template.format(vehicle_ids=vehicle_ids_str)

        # Build ChromaDB where filter
        where_filter: dict[str, Any] | None = None
        if supporting_ids:
            where_filter = {"unit_id": {"$in": supporting_ids}}

        return self._query_collection(
            query_string=query_string,
            n_results=self._top_k,
            where_filter=where_filter,
        )

    def retrieve_for_vehicle(
        self,
        unit_id: int,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve all events for a specific engine unit.

        Designed for the per-vehicle drill-down panel in the Streamlit UI.

        Args:
            unit_id: Integer engine unit identifier.
            top_k:   Maximum number of events to return.

        Returns:
            List of event dicts sorted by ``relevance_score`` descending.
        """
        query_string = (
            f"health events degradation maintenance engine {unit_id}"
        )
        where_filter: dict[str, Any] = {"unit_id": {"$eq": unit_id}}

        return self._query_collection(
            query_string=query_string,
            n_results=top_k,
            where_filter=where_filter,
        )

    def format_evidence_display(
        self,
        retrieved_events: list[dict[str, Any]],
    ) -> str:
        """Format retrieved events as a clean, human-readable string.

        Suitable for display in Streamlit (via ``st.text``) or as additional
        context passed to Gemini for synthesis.

        Args:
            retrieved_events: List of event dicts as returned by
                :meth:`retrieve` or :meth:`retrieve_for_vehicle`.

        Returns:
            Multi-line formatted string with one section per event.
        """
        if not retrieved_events:
            return "No specific events found for this query."

        lines: list[str] = []
        for rank, ev in enumerate(retrieved_events, 1):
            lines.append(
                f"[{rank}] {ev.get('event_type', 'event').upper()} · "
                f"Engine {ev.get('unit_id')} · "
                f"Cycle {ev.get('cycle')} · "
                f"Relevance: {ev.get('relevance_score', 0):.2f}"
            )
            lines.append(f"    {ev.get('description', '')}")
            lines.append(
                f"    Health: {ev.get('health_score_after', 'N/A')} · "
                f"Tier: {ev.get('risk_tier', 'N/A')}"
            )
            lines.append("")
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _query_collection(
        self,
        query_string: str,
        n_results: int,
        where_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Execute a ChromaDB semantic query and format results.

        Args:
            query_string: Natural-language query text.
            n_results:    Maximum number of results to request from ChromaDB.
            where_filter: Optional ChromaDB metadata filter dict.

        Returns:
            Filtered and formatted list of event dicts.
        """
        total_events = self._collection.count()
        if total_events == 0:
            logger.warning("ChromaDB collection is empty — no events to retrieve")
            return []

        # Cap n_results at the actual collection size to prevent ChromaDB error
        n_results = min(n_results, total_events)

        # Compute query embedding
        query_embedding = self._embedder.encode(
            [query_string], show_progress_bar=False
        ).tolist()

        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": query_embedding,
                "n_results":        n_results,
                "include":          ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter

            result = self._collection.query(**kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s — retrying without filter", exc)
            # Retry without the where filter (may happen if collection is small)
            result = self._collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        docs      = (result.get("documents")  or [[]])[0]
        metas     = (result.get("metadatas")  or [[]])[0]
        distances = (result.get("distances")  or [[]])[0]

        formatted: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance → similarity score: 1 - distance
            relevance = round(max(0.0, 1.0 - float(dist)), 4)
            if relevance < self._relevance_threshold:
                continue

            formatted.append({
                "event_id":          meta.get("event_id", ""),
                "unit_id":           int(meta.get("unit_id", 0)),
                "cycle":             int(meta.get("cycle", 0)),
                "event_type":        str(meta.get("event_type", "")),
                "description":       doc,
                "relevance_score":   relevance,
                "health_score_after": float(meta.get("health_score_after", 0)),
                "risk_tier":         str(meta.get("risk_tier", "")),
                "severity":          float(meta.get("severity", 0)),
            })

        formatted.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(
            "Retrieved %d events (above threshold=%.2f)  query_type=%s",
            len(formatted),
            self._relevance_threshold,
            where_filter,
        )
        return formatted
