"""
Apex Harness — Memory System (Cognee Graph Memory Integration).

Provides:
- Cross-session persistent memory with entity resolution
- Multi-strategy retrieval: semantic, keyword, and graph-based
- Memory CRUD operations with automatic entity extraction
- Session-aware recall with per-user weighting
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MemoryTier(Enum):
    """Memory hierarchy tiers."""

    TIER_1_COGNEE = "tier_1_cognee"      # Primary graph memory
    TIER_2_HINDSIGHT = "tier_2_hindsight"  # Cross-session retrieval
    TIER_3_FLAT = "tier_3_flat"          # Built-in flat file fallback


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class MemoryEntry:
    """A single memory entry stored in the system."""

    id: str
    content: str
    user_id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    tier: MemoryTier = MemoryTier.TIER_1_COGNEE
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    relevance_score: float = 0.5
    ttl_seconds: float | None = None  # None = no expiry

    def is_expired(self, now: float | None = None) -> bool:
        """Check if this entry has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        now = now or time.time()
        return (now - self.timestamp) > self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()

    def compute_hash(self) -> str:
        """Compute a deterministic hash for deduplication."""
        raw = f"{self.user_id}:{self.content}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class RetrievalResult:
    """Result from a memory retrieval operation."""

    entries: list[MemoryEntry]
    strategy_used: RetrievalStrategy
    total_candidates: int
    latency_ms: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EntityResolution:
    """Result of entity resolution — maps surface forms to canonical entities."""

    canonical: str
    surface_forms: list[str]
    entity_type: str
    confidence: float
    related_entities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity Resolution
# ---------------------------------------------------------------------------


class EntityResolver:
    """Resolves entities across sessions using fuzzy matching and aliases."""

    def __init__(self) -> None:
        self._canonical: dict[str, EntityResolution] = {}
        self._surface_index: dict[str, str] = {}  # surface form -> canonical
        self._alias_map: dict[str, list[str]] = {}

    def register_entity(
        self,
        canonical: str,
        entity_type: str = "unknown",
        aliases: list[str] | None = None,
        related: list[str] | None = None,
    ) -> None:
        """Register a canonical entity with optional aliases."""
        surface_aliases = aliases or []
        all_surfaces = [canonical.lower()] + [a.lower() for a in surface_aliases]

        resolution = EntityResolution(
            canonical=canonical,
            surface_forms=all_surfaces,
            entity_type=entity_type,
            confidence=1.0,
            related_entities=related or [],
        )
        self._canonical[canonical.lower()] = resolution
        for surface in all_surfaces:
            self._surface_index[surface] = canonical.lower()

    def resolve(self, text: str) -> list[EntityResolution]:
        """Find all entities mentioned in the given text."""
        text_lower = text.lower()
        found: dict[str, EntityResolution] = {}

        for surface, canonical_key in self._surface_index.items():
            if surface in text_lower:
                if canonical_key not in found:
                    found[canonical_key] = self._canonical[canonical_key]

        return list(found.values())

    def resolve_single(self, text: str) -> EntityResolution | None:
        """Resolve the single best entity match in text."""
        results = self.resolve(text)
        if not results:
            return None
        return max(results, key=lambda r: r.confidence)

    def add_alias(self, canonical: str, alias: str) -> None:
        """Add an alias to an existing canonical entity."""
        key = canonical.lower()
        if key not in self._canonical:
            self.register_entity(key, aliases=[alias])
            return
        resolution = self._canonical[key]
        resolution.surface_forms.append(alias.lower())
        self._surface_index[alias.lower()] = key

    def get_related(self, canonical: str) -> list[str]:
        """Get entities related to the given canonical entity."""
        key = canonical.lower()
        if key in self._canonical:
            return self._canonical[key].related_entities
        return []


# ---------------------------------------------------------------------------
# Retrieval Strategies
# ---------------------------------------------------------------------------


class RetrievalBackend(ABC):
    """Abstract base for retrieval backends."""

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return (entry, score) tuples sorted by relevance."""
        ...


class SemanticRetrieval(RetrievalBackend):
    """Semantic similarity-based retrieval using embeddings."""

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        self._embed_fn = embed_fn or self._default_embed
        self._store: dict[str, MemoryEntry] = {}

    @staticmethod
    def _default_embed(text: str) -> list[float]:
        """Simple bag-of-words embedding fallback (384-dim hashed)."""
        import re

        words = re.findall(r"\w+", text.lower())
        dim = 384
        vec = [0.0] * dim
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def index(self, entry: MemoryEntry) -> None:
        """Index a memory entry for semantic search."""
        entry.embedding = self._embed_fn(entry.content)
        self._store[entry.id] = entry

    def remove(self, entry_id: str) -> None:
        """Remove an entry from the index."""
        self._store.pop(entry_id, None)

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search by semantic similarity."""
        query_vec = self._embed_fn(query)
        results: list[tuple[MemoryEntry, float]] = []

        for entry in self._store.values():
            if entry.user_id != user_id:
                continue
            if entry.embedding is None:
                entry.embedding = self._embed_fn(entry.content)
            score = self._cosine_similarity(query_vec, entry.embedding)
            if score > 0:
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class KeywordRetrieval(RetrievalBackend):
    """BM25-inspired keyword retrieval."""

    def __init__(self) -> None:
        self._store: dict[str, MemoryEntry] = {}
        self._doc_freq: dict[str, int] = {}
        self._total_docs: int = 0

    def index(self, entry: MemoryEntry) -> None:
        """Index a memory entry for keyword search."""
        self._store[entry.id] = entry
        self._total_docs += 1
        words = set(self._tokenize(entry.content))
        for word in words:
            self._doc_freq[word] = self._doc_freq.get(word, 0) + 1

    def remove(self, entry_id: str) -> None:
        """Remove an entry from the index."""
        entry = self._store.pop(entry_id, None)
        if entry:
            self._total_docs -= 1
            words = set(self._tokenize(entry.content))
            for word in words:
                self._doc_freq[word] = max(0, self._doc_freq.get(word, 1) - 1)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer."""
        import re

        return re.findall(r"\w+", text.lower())

    def _bm25_score(
        self,
        query_terms: list[str],
        entry: MemoryEntry,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """Compute BM25 score for a document."""
        import math

        doc_terms = self._tokenize(entry.content)
        doc_len = len(doc_terms)
        if doc_len == 0:
            return 0.0

        # Average document length
        avg_dl = sum(
            len(self._tokenize(e.content)) for e in self._store.values()
        ) / max(1, self._total_docs)

        score = 0.0
        term_freq: dict[str, int] = {}
        for t in doc_terms:
            term_freq[t] = term_freq.get(t, 0) + 1

        for qt in query_terms:
            tf = term_freq.get(qt, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(qt, 0)
            idf = math.log(1 + (self._total_docs - df + 0.5) / (df + 0.5))
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))
            score += idf * tf_norm

        return score

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search by keyword relevance."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        results: list[tuple[MemoryEntry, float]] = []
        for entry in self._store.values():
            if entry.user_id != user_id:
                continue
            score = self._bm25_score(query_terms, entry)
            if score > 0:
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class GraphRetrieval(RetrievalBackend):
    """Graph-based retrieval traversing entity relationships."""

    def __init__(self, entity_resolver: EntityResolver) -> None:
        self._resolver = entity_resolver
        self._store: dict[str, MemoryEntry] = {}
        self._entity_graph: dict[str, set[str]] = {}  # entity -> connected entities

    def index(self, entry: MemoryEntry) -> None:
        """Index a memory entry for graph search."""
        self._store[entry.id] = entry
        # Build entity graph from entry entities
        for entity in entry.entities:
            key = entity.lower()
            if key not in self._entity_graph:
                self._entity_graph[key] = set()
            for other in entry.entities:
                if other.lower() != key:
                    self._entity_graph[key].add(other.lower())

    def remove(self, entry_id: str) -> None:
        """Remove an entry from the index."""
        self._store.pop(entry_id, None)

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search by graph traversal from query entities."""
        # Resolve entities in the query
        query_entities = self._resolver.resolve(query)
        if not query_entities:
            return []

        # Collect all connected entities (1-hop and 2-hop)
        connected: set[str] = set()
        for qe in query_entities:
            canonical = qe.canonical.lower()
            connected.add(canonical)
            # 1-hop
            if canonical in self._entity_graph:
                connected.update(self._entity_graph[canonical])
                # 2-hop
                for neighbor in self._entity_graph[canonical]:
                    if neighbor in self._entity_graph:
                        connected.update(self._entity_graph[neighbor])

        # Score entries by entity overlap
        results: list[tuple[MemoryEntry, float]] = []
        for entry in self._store.values():
            if entry.user_id != user_id:
                continue
            entry_entities = {e.lower() for e in entry.entities}
            overlap = len(entry_entities & connected)
            if overlap > 0:
                score = overlap / max(len(entry_entities), len(connected), 1)
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


# ---------------------------------------------------------------------------
# Cognee Integration
# ---------------------------------------------------------------------------


class CogneeGraphMemory:
    """Integration with Cognee graph memory backend.

    Falls back to in-memory storage when Cognee is not available.
    """

    def __init__(self, use_cognee: bool = False, data_dir: str | None = None) -> None:
        self._use_cognee = use_cognee
        self._data_dir = Path(data_dir) if data_dir else None
        self._fallback_store: dict[str, MemoryEntry] = {}
        self._cognee_available = False

        if use_cognee:
            self._try_init_cognee()

    def _try_init_cognee(self) -> None:
        """Attempt to initialize Cognee; fall back gracefully."""
        try:
            import cognee  # type: ignore

            self._cognee = cognee
            self._cognee_available = True
            logger.info("Cognee graph memory initialized successfully")
        except ImportError:
            logger.warning(
                "Cognee not installed; using in-memory fallback. "
                "Install with: pip install cognee"
            )
            self._cognee_available = False

    def add(self, entry: MemoryEntry) -> bool:
        """Add a memory entry to Cognee (or fallback)."""
        if self._cognee_available:
            try:
                # Cognee API: add data and cognify
                self._cognee.add(entry.content, dataset_name=entry.user_id)
                return True
            except Exception as exc:
                logger.error("Cognee add failed, using fallback: %s", exc)
                self._cognee_available = False

        # Fallback: in-memory + optional disk persistence
        self._fallback_store[entry.id] = entry
        if self._data_dir:
            self._persist_entry(entry)
        return True

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search Cognee graph memory."""
        if self._cognee_available:
            try:
                results = self._cognee.search(query, user_id=user_id)
                entries: list[tuple[MemoryEntry, float]] = []
                for item in results[:limit]:
                    entry = MemoryEntry(
                        id=str(item.get("id", hashlib.md5(
                            item.get("content", "").encode()
                        ).hexdigest()[:16])),
                        content=item.get("content", ""),
                        user_id=user_id,
                        session_id=item.get("session_id", ""),
                        metadata=item.get("metadata", {}),
                    )
                    score = float(item.get("score", 0.5))
                    entries.append((entry, score))
                return entries
            except Exception as exc:
                logger.error("Cognee search failed, using fallback: %s", exc)
                self._cognee_available = False

        # Fallback: simple keyword match on fallback store
        return self._fallback_search(query, user_id, limit)

    def _fallback_search(
        self,
        query: str,
        user_id: str,
        limit: int,
    ) -> list[tuple[MemoryEntry, float]]:
        """Simple fallback search over in-memory store."""
        import re

        query_terms = set(re.findall(r"\w+", query.lower()))
        results: list[tuple[MemoryEntry, float]] = []

        for entry in self._fallback_store.values():
            if entry.user_id != user_id:
                continue
            content_terms = set(re.findall(r"\w+", entry.content.lower()))
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _persist_entry(self, entry: MemoryEntry) -> None:
        """Persist an entry to disk as JSON."""
        if not self._data_dir:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        path = self._data_dir / f"{entry.id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": entry.id,
                    "content": entry.content,
                    "user_id": entry.user_id,
                    "session_id": entry.session_id,
                    "timestamp": entry.timestamp,
                    "entities": entry.entities,
                    "metadata": entry.metadata,
                    "tier": entry.tier.value,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed,
                    "relevance_score": entry.relevance_score,
                    "ttl_seconds": entry.ttl_seconds,
                },
                indent=2,
            )
        )

    def prune(self, user_id: str | None = None) -> int:
        """Remove expired entries. Returns count pruned."""
        to_remove: list[str] = []
        now = time.time()
        for eid, entry in self._fallback_store.items():
            if user_id and entry.user_id != user_id:
                continue
            if entry.is_expired(now):
                to_remove.append(eid)

        for eid in to_remove:
            del self._fallback_store[eid]
            if self._data_dir:
                fpath = self._data_dir / f"{eid}.json"
                if fpath.exists():
                    fpath.unlink()

        return len(to_remove)


# ---------------------------------------------------------------------------
# Multi-Strategy Memory Manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Unified memory manager with multi-strategy retrieval.

    Orchestrates semantic, keyword, and graph retrieval backends,
    entity resolution, and cross-session persistence.
    """

    def __init__(
        self,
        use_cognee: bool = False,
        data_dir: str | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._resolver = EntityResolver()
        self._semantic = SemanticRetrieval(embed_fn=embed_fn)
        self._keyword = KeywordRetrieval()
        self._graph = GraphRetrieval(self._resolver)
        self._cognee = CogneeGraphMemory(use_cognee=use_cognee, data_dir=data_dir)
        self._all_entries: dict[str, MemoryEntry] = {}
        self._user_entries: dict[str, set[str]] = {}  # user_id -> entry_ids
        self._session_entries: dict[str, set[str]] = {}  # session_id -> entry_ids

    # -- CRUD Operations ---------------------------------------------------

    def add(
        self,
        content: str,
        user_id: str,
        session_id: str = "",
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> MemoryEntry:
        """Add a new memory entry.

        Automatically extracts entities if not provided, resolves them,
        and indexes across all retrieval backends.
        """
        entry_id = hashlib.sha256(
            f"{user_id}:{session_id}:{content}:{time.time()}".encode()
        ).hexdigest()[:16]

        # Auto-extract entities if not provided
        if entities is None:
            resolved = self._resolver.resolve(content)
            entities = [r.canonical for r in resolved]

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            user_id=user_id,
            session_id=session_id,
            entities=entities,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
        )

        # Index in all backends
        self._semantic.index(entry)
        self._keyword.index(entry)
        self._graph.index(entry)
        self._cognee.add(entry)

        # Track in manager
        self._all_entries[entry_id] = entry
        self._user_entries.setdefault(user_id, set()).add(entry_id)
        if session_id:
            self._session_entries.setdefault(session_id, set()).add(entry_id)

        return entry

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by ID."""
        entry = self._all_entries.get(entry_id)
        if entry:
            entry.touch()
        return entry

    def update(self, entry_id: str, **kwargs: Any) -> MemoryEntry | None:
        """Update fields on an existing entry."""
        entry = self._all_entries.get(entry_id)
        if not entry:
            return None

        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        # Re-index if content changed
        if "content" in kwargs:
            self._semantic.remove(entry_id)
            self._keyword.remove(entry_id)
            self._graph.remove(entry_id)
            self._semantic.index(entry)
            self._keyword.index(entry)
            self._graph.index(entry)

        return entry

    def delete(self, entry_id: str) -> bool:
        """Delete an entry from all backends."""
        entry = self._all_entries.pop(entry_id, None)
        if not entry:
            return False

        self._semantic.remove(entry_id)
        self._keyword.remove(entry_id)
        self._graph.remove(entry_id)

        # Remove from user/session indexes
        user_set = self._user_entries.get(entry.user_id)
        if user_set:
            user_set.discard(entry_id)
        if entry.session_id:
            session_set = self._session_entries.get(entry.session_id)
            if session_set:
                session_set.discard(entry_id)

        return True

    # -- Retrieval ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        limit: int = 10,
        session_id: str | None = None,
    ) -> RetrievalResult:
        """Retrieve memories using the specified strategy.

        HYBRID merges results from all backends with reciprocal rank fusion.
        """
        start = time.time()

        if strategy == RetrievalStrategy.SEMANTIC:
            raw = self._semantic.search(query, user_id, limit)
        elif strategy == RetrievalStrategy.KEYWORD:
            raw = self._keyword.search(query, user_id, limit)
        elif strategy == RetrievalStrategy.GRAPH:
            raw = self._graph.search(query, user_id, limit)
        elif strategy == RetrievalStrategy.HYBRID:
            raw = self._hybrid_search(query, user_id, limit)
        else:
            raw = self._hybrid_search(query, user_id, limit)

        # Filter by session if specified
        if session_id:
            raw = [(e, s) for e, s in raw if e.session_id == session_id]

        # Update access metadata
        entries: list[MemoryEntry] = []
        scores: dict[str, float] = {}
        for entry, score in raw:
            entry.touch()
            entries.append(entry)
            scores[entry.id] = score

        latency = (time.time() - start) * 1000

        return RetrievalResult(
            entries=entries,
            strategy_used=strategy,
            total_candidates=len(self._user_entries.get(user_id, set())),
            latency_ms=latency,
            scores=scores,
        )

    def _hybrid_search(
        self,
        query: str,
        user_id: str,
        limit: int,
        k: int = 60,
    ) -> list[tuple[MemoryEntry, float]]:
        """Reciprocal Rank Fusion across all backends."""
        all_results: dict[str, list[tuple[MemoryEntry, float]]] = {
            "semantic": self._semantic.search(query, user_id, limit * 2),
            "keyword": self._keyword.search(query, user_id, limit * 2),
            "graph": self._graph.search(query, user_id, limit * 2),
        }

        # RRF scoring
        rrf_scores: dict[str, float] = {}
        entry_map: dict[str, MemoryEntry] = {}

        for backend_results in all_results.values():
            for rank, (entry, _score) in enumerate(backend_results):
                eid = entry.id
                rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
                entry_map[eid] = entry

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [(entry_map[eid], rrf_scores[eid]) for eid in sorted_ids[:limit]]

    # -- Entity Management -------------------------------------------------

    def register_entity(
        self,
        canonical: str,
        entity_type: str = "unknown",
        aliases: list[str] | None = None,
        related: list[str] | None = None,
    ) -> None:
        """Register an entity for resolution."""
        self._resolver.register_entity(canonical, entity_type, aliases, related)

    def resolve_entities(self, text: str) -> list[EntityResolution]:
        """Resolve entities in text."""
        return self._resolver.resolve(text)

    # -- Session & User Management ------------------------------------------

    def get_session_memories(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Get all memories for a specific session."""
        entry_ids = self._session_entries.get(session_id, set())
        results: list[MemoryEntry] = []
        for eid in entry_ids:
            entry = self._all_entries.get(eid)
            if entry and (user_id is None or entry.user_id == user_id):
                results.append(entry)
        return sorted(results, key=lambda e: e.timestamp)

    def get_user_stats(self, user_id: str) -> dict[str, Any]:
        """Get memory statistics for a user."""
        entry_ids = self._user_entries.get(user_id, set())
        entries = [self._all_entries[eid] for eid in entry_ids if eid in self._all_entries]
        if not entries:
            return {"total": 0, "sessions": 0, "entities": 0}

        sessions = {e.session_id for e in entries if e.session_id}
        all_entities: set[str] = set()
        for e in entries:
            all_entities.update(e.entities)

        return {
            "total": len(entries),
            "sessions": len(sessions),
            "entities": len(all_entities),
            "avg_relevance": sum(e.relevance_score for e in entries) / len(entries),
            "oldest": min(e.timestamp for e in entries),
            "newest": max(e.timestamp for e in entries),
        }

    # -- Maintenance -------------------------------------------------------

    def prune_expired(self, user_id: str | None = None) -> int:
        """Remove expired entries. Returns count pruned."""
        to_remove: list[str] = []
        now = time.time()

        for eid, entry in self._all_entries.items():
            if user_id and entry.user_id != user_id:
                continue
            if entry.is_expired(now):
                to_remove.append(eid)

        for eid in to_remove:
            self.delete(eid)

        # Also prune Cognee fallback
        self._cognee.prune(user_id)

        return len(to_remove)

    def clear_user(self, user_id: str) -> int:
        """Clear all memories for a user. Returns count removed."""
        entry_ids = list(self._user_entries.get(user_id, set()))
        for eid in entry_ids:
            self.delete(eid)
        self._user_entries.pop(user_id, None)
        return len(entry_ids)
