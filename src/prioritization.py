"""
Apex Harness — Memory Prioritization & Garbage Collection.

Provides:
- Memory GC with TTL, LRU, LFU, staleness, and relevance eviction
- Priority scoring with P0-P4 tiers
- Personalization with per-user memory weighting
- Configurable eviction policies and scheduling
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class PriorityTier(Enum):
    """Memory priority tiers."""

    P0_CRITICAL = "p0_critical"    # score >= 0.8
    P1_HIGH = "p1_high"            # score 0.6-0.8
    P2_MEDIUM = "p2_medium"        # score 0.4-0.6
    P3_LOW = "p3_low"              # score 0.2-0.4
    P4_DEAD = "p4_dead"            # score < 0.2


class EvictionReason(Enum):
    """Reasons for memory eviction."""

    TTL_EXPIRED = "ttl_expired"
    LRU_EVICTED = "lru_evicted"
    LFU_EVICTED = "lfu_evicted"
    STALE = "stale"
    LOW_RELEVANCE = "low_relevance"
    CAPACITY = "capacity"
    MANUAL = "manual"


@dataclass
class PriorityScore:
    """Composite priority score with component breakdown."""

    total: float                            # 0.0 - 1.0
    relevance: float = 0.0                  # semantic relevance
    recency: float = 0.0                    # time-based decay
    frequency: float = 0.0                  # access frequency
    user_weight: float = 0.0                # per-user personalization
    staleness_penalty: float = 0.0          # penalty for staleness
    tier: PriorityTier = PriorityTier.P4_DEAD

    def __post_init__(self) -> None:
        self.tier = self._compute_tier(self.total)

    @staticmethod
    def _compute_tier(score: float) -> PriorityTier:
        if score >= 0.8:
            return PriorityTier.P0_CRITICAL
        elif score >= 0.6:
            return PriorityTier.P1_HIGH
        elif score >= 0.4:
            return PriorityTier.P2_MEDIUM
        elif score >= 0.2:
            return PriorityTier.P3_LOW
        else:
            return PriorityTier.P4_DEAD


@dataclass
class EvictionRecord:
    """Record of a memory eviction event."""

    memory_id: str
    user_id: str
    reason: EvictionReason
    score_at_eviction: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Per-user personalization profile."""

    user_id: str
    weights: dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.30,
        "recency": 0.25,
        "frequency": 0.20,
        "user_affinity": 0.15,
        "diversity": 0.10,
    })
    affinity_entities: dict[str, float] = field(default_factory=dict)
    affinity_sessions: dict[str, float] = field(default_factory=dict)
    total_interactions: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def update_weight(self, component: str, weight: float) -> None:
        """Update a single weight component (auto-normalizes)."""
        self.weights[component] = max(0.0, min(1.0, weight))
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total

    def record_interaction(self, entity: str = "", session_id: str = "") -> None:
        """Record a user interaction for affinity tracking."""
        self.total_interactions += 1
        self.last_active = time.time()
        if entity:
            self.affinity_entities[entity] = (
                self.affinity_entities.get(entity, 0.0) + 1.0
            )
        if session_id:
            self.affinity_sessions[session_id] = (
                self.affinity_sessions.get(session_id, 0.0) + 1.0
            )

    def get_affinity(self, entity: str) -> float:
        """Get normalized affinity score for an entity."""
        if not self.affinity_entities:
            return 0.0
        max_affinity = max(self.affinity_entities.values())
        if max_affinity == 0:
            return 0.0
        return self.affinity_entities.get(entity, 0.0) / max_affinity


# ---------------------------------------------------------------------------
# Eviction Policies
# ---------------------------------------------------------------------------


class EvictionPolicy(ABC):
    """Abstract base for eviction policies."""

    @abstractmethod
    def select_victims(
        self,
        entries: list[Any],
        count: int,
        now: float | None = None,
    ) -> list[tuple[Any, EvictionReason]]:
        """Select entries to evict. Returns (entry, reason) tuples."""
        ...


class TTLEvictionPolicy(EvictionPolicy):
    """Evict entries that have exceeded their time-to-live."""

    def select_victims(
        self,
        entries: list[Any],
        count: int,
        now: float | None = None,
    ) -> list[tuple[Any, EvictionReason]]:
        now = now or time.time()
        victims: list[tuple[Any, EvictionReason]] = []
        for entry in entries:
            if len(victims) >= count:
                break
            if entry.is_expired(now):
                victims.append((entry, EvictionReason.TTL_EXPIRED))
        return victims


class LRUEvictionPolicy(EvictionPolicy):
    """Evict least recently used entries."""

    def select_victims(
        self,
        entries: list[Any],
        count: int,
        now: float | None = None,
    ) -> list[tuple[Any, EvictionReason]]:
        sorted_entries = sorted(entries, key=lambda e: e.last_accessed)
        return [
            (entry, EvictionReason.LRU_EVICTED)
            for entry in sorted_entries[:count]
        ]


class LFUEvictionPolicy(EvictionPolicy):
    """Evict least frequently used entries."""

    def select_victims(
        self,
        entries: list[Any],
        count: int,
        now: float | None = None,
    ) -> list[tuple[Any, EvictionReason]]:
        sorted_entries = sorted(entries, key=lambda e: e.access_count)
        return [
            (entry, EvictionReason.LFU_EVICTED)
            for entry in sorted_entries[:count]
        ]


class StalenessEvictionPolicy(EvictionPolicy):
    """Evict entries that haven't been accessed within a staleness window."""

    def __init__(self, max_staleness_seconds: float = 86400.0) -> None:
        self._max_staleness = max_staleness_seconds

    def select_victims(
        self,
        entries: list[Any],
        count: int,
        now: float | None = None,
    ) -> list[tuple[Any, EvictionReason]]:
        now = now or time.time()
        victims: list[tuple[Any, EvictionReason]] = []
        for entry in entries:
            if len(victims) >= count:
                break
            if (now - entry.last_accessed) > self._max_staleness:
                victims.append((entry, EvictionReason.STALE))
        return victims


class RelevanceEvictionPolicy(EvictionPolicy):
    """Evict entries with the lowest relevance scores."""

    def __init__(self, threshold: float = 0.2) -> None:
        self._threshold = threshold

    def select_victims(
        self,
        entries: list[Any],
        count: int,
        now: float | None = None,
    ) -> list[tuple[Any, EvictionReason]]:
        sorted_entries = sorted(entries, key=lambda e: e.relevance_score)
        victims: list[tuple[Any, EvictionReason]] = []
        for entry in sorted_entries:
            if len(victims) >= count:
                break
            if entry.relevance_score < self._threshold:
                victims.append((entry, EvictionReason.LOW_RELEVANCE))
        return victims


class CompositeEvictionPolicy(EvictionPolicy):
    """Combines multiple eviction policies with priority ordering."""

    def __init__(self, policies: list[EvictionPolicy] | None = None) -> None:
        self._policies = policies or [
            TTLEvictionPolicy(),
            StalenessEvictionPolicy(),
            RelevanceEvictionPolicy(),
            LRUEvictionPolicy(),
        ]

    def select_victims(
        self,
        entries: list[Any],
        count: int,
        now: float | None = None,
    ) -> list[tuple[Any, EvictionReason]]:
        """Apply policies in order, deduplicating victims."""
        victims: list[tuple[Any, EvictionReason]] = []
        evicted_ids: set[str] = set()
        remaining = count

        for policy in self._policies:
            if remaining <= 0:
                break
            # Filter out already-evicted entries
            available = [
                e for e in entries
                if e.id not in evicted_ids
            ]
            selected = policy.select_victims(available, remaining, now)
            for entry, reason in selected:
                if entry.id not in evicted_ids:
                    victims.append((entry, reason))
                    evicted_ids.add(entry.id)
                    remaining -= 1

        return victims


# ---------------------------------------------------------------------------
# Priority Scoring
# ---------------------------------------------------------------------------


class PriorityScorer:
    """Computes priority scores for memory entries."""

    def __init__(
        self,
        decay_half_life: float = 86400.0,  # 24 hours
        max_frequency: int = 100,
    ) -> None:
        self._decay_half_life = decay_half_life
        self._max_frequency = max_frequency

    def score(
        self,
        entry: Any,
        user_profile: UserProfile | None = None,
        query_embedding: list[float] | None = None,
        now: float | None = None,
    ) -> PriorityScore:
        """Compute composite priority score for an entry."""
        now = now or time.time()

        # Relevance component
        relevance = entry.relevance_score

        # Recency component (exponential decay)
        age = now - entry.timestamp
        recency = math.exp(-age * math.log(2) / self._decay_half_life)

        # Frequency component (log-scaled)
        frequency = math.log1p(entry.access_count) / math.log1p(self._max_frequency)

        # User affinity component
        user_weight = 0.0
        if user_profile:
            # Average affinity across entry entities
            if entry.entities:
                affinities = [
                    user_profile.get_affinity(e) for e in entry.entities
                ]
                user_weight = sum(affinities) / len(affinities)
            else:
                user_weight = 0.5  # neutral

        # Staleness penalty
        time_since_access = now - entry.last_accessed
        staleness = min(1.0, time_since_access / self._decay_half_life)
        staleness_penalty = staleness * 0.3  # up to 30% penalty

        # Weighted combination
        if user_profile:
            w = user_profile.weights
            total = (
                w.get("relevance", 0.3) * relevance
                + w.get("recency", 0.25) * recency
                + w.get("frequency", 0.2) * frequency
                + w.get("user_affinity", 0.15) * user_weight
            )
        else:
            total = 0.35 * relevance + 0.30 * recency + 0.20 * frequency + 0.15 * user_weight

        # Apply staleness penalty
        total = max(0.0, total - staleness_penalty)

        return PriorityScore(
            total=total,
            relevance=relevance,
            recency=recency,
            frequency=frequency,
            user_weight=user_weight,
            staleness_penalty=staleness_penalty,
        )


# ---------------------------------------------------------------------------
# Memory Garbage Collector
# ---------------------------------------------------------------------------


class MemoryGC:
    """Garbage collector for memory entries with configurable policies."""

    def __init__(
        self,
        max_entries: int = 10000,
        eviction_policy: EvictionPolicy | None = None,
        scorer: PriorityScorer | None = None,
        auto_gc_threshold: float = 0.9,  # Trigger GC at 90% capacity
    ) -> None:
        self._max_entries = max_entries
        self._eviction_policy = eviction_policy or CompositeEvictionPolicy()
        self._scorer = scorer or PriorityScorer()
        self._auto_gc_threshold = auto_gc_threshold
        self._eviction_log: list[EvictionRecord] = []
        self._user_profiles: dict[str, UserProfile] = {}
        self._gc_runs: int = 0
        self._last_gc: float | None = None

    def register_user(self, user_id: str, weights: dict[str, float] | None = None) -> UserProfile:
        """Register a user profile for personalization."""
        profile = UserProfile(user_id=user_id, weights=weights or {})
        self._user_profiles[user_id] = profile
        return profile

    def get_profile(self, user_id: str) -> UserProfile | None:
        """Get a user profile."""
        return self._user_profiles.get(user_id)

    def update_profile_weight(
        self,
        user_id: str,
        component: str,
        weight: float,
    ) -> bool:
        """Update a weight component for a user."""
        profile = self._user_profiles.get(user_id)
        if not profile:
            return False
        profile.update_weight(component, weight)
        return True

    def record_interaction(
        self,
        user_id: str,
        entity: str = "",
        session_id: str = "",
    ) -> None:
        """Record a user interaction for affinity tracking."""
        profile = self._user_profiles.get(user_id)
        if profile:
            profile.record_interaction(entity, session_id)

    def should_gc(self, current_count: int) -> bool:
        """Check if GC should run based on capacity threshold."""
        return current_count >= (self._max_entries * self._auto_gc_threshold)

    def run_gc(
        self,
        entries: list[Any],
        user_id: str | None = None,
        target_count: int | None = None,
        now: float | None = None,
    ) -> list[EvictionRecord]:
        """Run garbage collection on memory entries.

        Returns list of eviction records.
        """
        now = now or time.time()
        self._gc_runs += 1
        self._last_gc = now

        # Filter by user if specified
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]

        # Determine how many to evict
        if target_count is not None:
            to_evict = max(0, len(entries) - target_count)
        else:
            to_evict = max(0, len(entries) - int(self._max_entries * 0.8))

        if to_evict == 0:
            return []

        # Select victims
        victims = self._eviction_policy.select_victims(entries, to_evict, now)

        # Score and log
        records: list[EvictionRecord] = []
        profile = self._user_profiles.get(user_id) if user_id else None

        for entry, reason in victims:
            score = self._scorer.score(entry, profile, now=now)
            record = EvictionRecord(
                memory_id=entry.id,
                user_id=entry.user_id,
                reason=reason,
                score_at_eviction=score.total,
                metadata={
                    "tier": score.tier.value,
                    "relevance": score.relevance,
                    "recency": score.recency,
                    "frequency": score.frequency,
                },
            )
            records.append(record)
            self._eviction_log.append(record)

        return records

    def get_priority_distribution(
        self,
        entries: list[Any],
        user_id: str | None = None,
    ) -> dict[PriorityTier, int]:
        """Get distribution of entries across priority tiers."""
        profile = self._user_profiles.get(user_id) if user_id else None
        distribution: dict[PriorityTier, int] = {tier: 0 for tier in PriorityTier}

        for entry in entries:
            score = self._scorer.score(entry, profile)
            distribution[score.tier] += 1

        return distribution

    def get_top_k(
        self,
        entries: list[Any],
        k: int,
        user_id: str | None = None,
    ) -> list[tuple[Any, PriorityScore]]:
        """Get top-k entries by priority score."""
        profile = self._user_profiles.get(user_id) if user_id else None
        scored: list[tuple[float, Any, PriorityScore]] = []

        for entry in entries:
            score = self._scorer.score(entry, profile)
            scored.append((score.total, entry, score))

        # Use heapq for efficient top-k
        top = heapq.nlargest(k, scored, key=lambda x: x[0])
        return [(entry, score) for _, entry, score in top]

    def get_eviction_log(
        self,
        user_id: str | None = None,
        reason: EvictionReason | None = None,
    ) -> list[EvictionRecord]:
        """Query eviction log with optional filters."""
        results = self._eviction_log
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        if reason:
            results = [r for r in results if r.reason == reason]
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get GC statistics."""
        by_reason: dict[str, int] = {}
        for record in self._eviction_log:
            key = record.reason.value
            by_reason[key] = by_reason.get(key, 0) + 1

        return {
            "gc_runs": self._gc_runs,
            "last_gc": self._last_gc,
            "total_evicted": len(self._eviction_log),
            "by_reason": by_reason,
            "max_entries": self._max_entries,
            "registered_users": len(self._user_profiles),
        }


# ---------------------------------------------------------------------------
# Prioritization Manager
# ---------------------------------------------------------------------------


class PrioritizationManager:
    """Unified prioritization manager combining scoring, GC, and personalization."""

    def __init__(
        self,
        max_entries: int = 10000,
        decay_half_life: float = 86400.0,
        auto_gc_threshold: float = 0.9,
    ) -> None:
        self._gc = MemoryGC(
            max_entries=max_entries,
            scorer=PriorityScorer(decay_half_life=decay_half_life),
            auto_gc_threshold=auto_gc_threshold,
        )

    @property
    def gc(self) -> MemoryGC:
        return self._gc

    def register_user(
        self,
        user_id: str,
        weights: dict[str, float] | None = None,
    ) -> UserProfile:
        """Register a user for personalized prioritization."""
        return self._gc.register_user(user_id, weights)

    def score_memory(
        self,
        entry: Any,
        user_id: str | None = None,
    ) -> PriorityScore:
        """Score a single memory entry."""
        profile = self._gc.get_profile(user_id) if user_id else None
        return self._gc._scorer.score(entry, profile)

    def rank_memories(
        self,
        entries: list[Any],
        user_id: str | None = None,
        k: int | None = None,
    ) -> list[tuple[Any, PriorityScore]]:
        """Rank memories by priority for a user."""
        k = k or len(entries)
        return self._gc.get_top_k(entries, k, user_id)

    def maybe_gc(
        self,
        entries: list[Any],
        user_id: str | None = None,
    ) -> list[EvictionRecord]:
        """Run GC if capacity threshold is exceeded."""
        if self._gc.should_gc(len(entries)):
            return self._gc.run_gc(entries, user_id)
        return []

    def get_tier_counts(
        self,
        entries: list[Any],
        user_id: str | None = None,
    ) -> dict[str, int]:
        """Get count of entries per priority tier."""
        dist = self._gc.get_priority_distribution(entries, user_id)
        return {tier.value: count for tier, count in dist.items()}

    def personalize_weights(
        self,
        user_id: str,
        interactions: list[dict[str, Any]],
    ) -> UserProfile:
        """Auto-tune user weights based on interaction history.

        Analyzes which components correlate with positive outcomes
        and adjusts weights accordingly.
        """
        profile = self._gc.get_profile(user_id)
        if not profile:
            profile = self._gc.register_user(user_id)

        if not interactions:
            return profile

        # Simple heuristic: boost weights for components that
        # correlate with high-rated interactions
        component_totals: dict[str, float] = {
            "relevance": 0.0,
            "recency": 0.0,
            "frequency": 0.0,
            "user_affinity": 0.0,
        }
        count = 0

        for interaction in interactions:
            rating = interaction.get("rating", 0.5)
            for component in component_totals:
                component_totals[component] += rating * interaction.get(component, 0.5)
            count += 1

        if count > 0:
            # Normalize and update
            total = sum(component_totals.values())
            if total > 0:
                for component, value in component_totals.items():
                    new_weight = value / total
                    profile.update_weight(component, new_weight)

        return profile
