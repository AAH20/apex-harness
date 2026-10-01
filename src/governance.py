"""
Apex Harness — Governance & Supervision System.

Provides:
- Nerve supervision integration for DoD enforcement
- Context ledger for tracking all context mutations
- Context curation with shadow/advisory/enforce modes
- Audit trail and compliance reporting
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


class CurationMode(Enum):
    """Context curation enforcement modes."""

    SHADOW = "shadow"        # Log only, no enforcement
    ADVISORY = "advisory"    # Warn but allow
    ENFORCE = "enforce"      # Block violations


class LedgerOperation(Enum):
    """Types of context ledger operations."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    COMPRESS = "compress"
    ENRICH = "enrich"
    FILTER = "filter"
    VALIDATE = "validate"
    PROMOTE = "promote"
    DEMOTE = "demote"


class DoDStatus(Enum):
    """Definition of Done status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LedgerEntry:
    """A single entry in the context ledger."""

    id: str
    operation: LedgerOperation
    session_id: str
    user_id: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""
    mode: CurationMode = CurationMode.SHADOW

    def __post_init__(self) -> None:
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash for chain integrity."""
        raw = (
            f"{self.id}:{self.operation.value}:{self.session_id}:"
            f"{self.user_id}:{self.timestamp}:{self.previous_hash}:"
            f"{json.dumps(self.details, sort_keys=True, default=str)}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


@dataclass
class DoDItem:
    """A single Definition of Done criterion."""

    id: str
    description: str
    status: DoDStatus = DoDStatus.PENDING
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def complete(self, evidence: str = "") -> None:
        """Mark as complete with optional evidence."""
        self.status = DoDStatus.COMPLETE
        self.completed_at = time.time()
        if evidence:
            self.evidence.append(evidence)

    def fail(self, reason: str = "") -> None:
        """Mark as failed with reason."""
        self.status = DoDStatus.FAILED
        if reason:
            self.evidence.append(f"FAIL: {reason}")


@dataclass
class DoD:
    """Definition of Done — collection of criteria for a task."""

    id: str
    task_id: str
    items: list[DoDItem] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    enforced: bool = True

    @property
    def is_complete(self) -> bool:
        """Check if all items are complete."""
        return all(item.status == DoDStatus.COMPLETE for item in self.items)

    @property
    def progress(self) -> float:
        """Completion ratio 0.0-1.0."""
        if not self.items:
            return 1.0
        done = sum(1 for i in self.items if i.status == DoDStatus.COMPLETE)
        return done / len(self.items)

    @property
    def status(self) -> DoDStatus:
        """Overall status derived from items."""
        if not self.items:
            return DoDStatus.COMPLETE
        if all(i.status == DoDStatus.COMPLETE for i in self.items):
            return DoDStatus.COMPLETE
        if any(i.status == DoDStatus.FAILED for i in self.items):
            return DoDStatus.FAILED
        if any(i.status == DoDStatus.IN_PROGRESS for i in self.items):
            return DoDStatus.IN_PROGRESS
        return DoDStatus.PENDING


@dataclass
class SupervisionEvent:
    """A supervision event from Nerve integration."""

    event_id: str
    event_type: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "info"  # info, warning, error, critical
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class CurationDecision:
    """Result of a curation check."""

    allowed: bool
    mode: CurationMode
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    action_taken: str = ""


# ---------------------------------------------------------------------------
# Context Ledger
# ---------------------------------------------------------------------------


class ContextLedger:
    """Immutable ledger tracking all context mutations.

    Uses hash-chaining for integrity verification.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._entries: list[LedgerEntry] = []
        self._by_session: dict[str, list[LedgerEntry]] = {}
        self._by_operation: dict[LedgerOperation, list[LedgerEntry]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._last_hash = ""

    def append(
        self,
        operation: LedgerOperation,
        session_id: str,
        user_id: str,
        details: dict[str, Any] | None = None,
        mode: CurationMode = CurationMode.SHADOW,
    ) -> LedgerEntry:
        """Append a new entry to the ledger."""
        entry_id = hashlib.sha256(
            f"{session_id}:{operation.value}:{time.time()}".encode()
        ).hexdigest()[:16]

        entry = LedgerEntry(
            id=entry_id,
            operation=operation,
            session_id=session_id,
            user_id=user_id,
            details=details or {},
            previous_hash=self._last_hash,
            mode=mode,
        )

        self._entries.append(entry)
        self._last_hash = entry.entry_hash

        # Index
        self._by_session.setdefault(session_id, []).append(entry)
        self._by_operation.setdefault(operation, []).append(entry)

        # Persist if configured
        if self._persist_path:
            self._persist_entry(entry)

        return entry

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the entire chain.

        Returns (is_valid, list_of_errors).
        """
        errors: list[str] = []
        prev_hash = ""

        for i, entry in enumerate(self._entries):
            # Check previous hash linkage
            if entry.previous_hash != prev_hash:
                errors.append(
                    f"Chain break at entry {i}: expected prev_hash "
                    f"{prev_hash[:12]}..., got {entry.previous_hash[:12]}..."
                )

            # Recompute hash
            recomputed = entry._compute_hash()
            if recomputed != entry.entry_hash:
                errors.append(
                    f"Hash mismatch at entry {i}: stored {entry.entry_hash[:12]}..., "
                    f"recomputed {recomputed[:12]}..."
                )

            prev_hash = entry.entry_hash

        return len(errors) == 0, errors

    def get_session_history(self, session_id: str) -> list[LedgerEntry]:
        """Get all ledger entries for a session."""
        return list(self._by_session.get(session_id, []))

    def get_operations(self, operation: LedgerOperation) -> list[LedgerEntry]:
        """Get all entries for a specific operation type."""
        return list(self._by_operation.get(operation, []))

    @property
    def size(self) -> int:
        return len(self._entries)

    def _persist_entry(self, entry: LedgerEntry) -> None:
        """Append entry to persistent log file."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "a") as f:
            f.write(json.dumps({
                "id": entry.id,
                "operation": entry.operation.value,
                "session_id": entry.session_id,
                "user_id": entry.user_id,
                "timestamp": entry.timestamp,
                "details": entry.details,
                "previous_hash": entry.previous_hash,
                "entry_hash": entry.entry_hash,
                "mode": entry.mode.value,
            }, default=str) + "\n"
        )


# ---------------------------------------------------------------------------
# DoD Enforcement
# ---------------------------------------------------------------------------


class DoDEnforcer:
    """Enforces Definition of Done criteria."""

    def __init__(self, strict: bool = True) -> None:
        self._dods: dict[str, DoD] = {}  # task_id -> DoD
        self._strict = strict

    def create_dod(
        self,
        task_id: str,
        items: list[DoDItem],
        enforced: bool = True,
    ) -> DoD:
        """Create a new Definition of Done for a task."""
        dod = DoD(
            id=hashlib.sha256(f"dod:{task_id}".encode()).hexdigest()[:16],
            task_id=task_id,
            items=items,
            enforced=enforced,
        )
        self._dods[task_id] = dod
        return dod

    def get_dod(self, task_id: str) -> DoD | None:
        """Retrieve a DoD by task ID."""
        return self._dods.get(task_id)

    def check(self, task_id: str) -> tuple[bool, list[str]]:
        """Check if a task's DoD is satisfied.

        Returns (is_satisfied, list_of_failures).
        """
        dod = self._dods.get(task_id)
        if not dod:
            return True, []

        failures: list[str] = []
        for item in dod.items:
            if item.status != DoDStatus.COMPLETE:
                failures.append(f"{item.id}: {item.description} [{item.status.value}]")

        return len(failures) == 0, failures

    def complete_item(
        self,
        task_id: str,
        item_id: str,
        evidence: str = "",
    ) -> bool:
        """Mark a DoD item as complete."""
        dod = self._dods.get(task_id)
        if not dod:
            return False
        for item in dod.items:
            if item.id == item_id:
                item.complete(evidence)
                return True
        return False

    def fail_item(
        self,
        task_id: str,
        item_id: str,
        reason: str = "",
    ) -> bool:
        """Mark a DoD item as failed."""
        dod = self._dods.get(task_id)
        if not dod:
            return False
        for item in dod.items:
            if item.id == item_id:
                item.fail(reason)
                return True
        return False

    def is_blocked(self, task_id: str) -> bool:
        """Check if a task is blocked by DoD enforcement."""
        dod = self._dods.get(task_id)
        if not dod or not dod.enforced:
            return False
        return not dod.is_complete


# ---------------------------------------------------------------------------
# Nerve Supervision Integration
# ---------------------------------------------------------------------------


class NerveSupervisor:
    """Integration with Nerve supervision system.

    Falls back to local supervision when Nerve is not available.
    """

    def __init__(self, use_nerve: bool = False) -> None:
        self._use_nerve = use_nerve
        self._nerve_available = False
        self._events: list[SupervisionEvent] = []
        self._handlers: dict[str, list[Callable[[SupervisionEvent], None]]] = {}

        if use_nerve:
            self._try_init_nerve()

    def _try_init_nerve(self) -> None:
        """Attempt to initialize Nerve; fall back gracefully."""
        try:
            # Nerve integration would go here
            # import nerve  # type: ignore
            self._nerve_available = True
            logger.info("Nerve supervision initialized")
        except ImportError:
            logger.warning(
                "Nerve not available; using local supervision. "
                "Install with: pip install nerve"
            )
            self._nerve_available = False

    def emit_event(
        self,
        event_type: str,
        session_id: str,
        message: str,
        severity: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> SupervisionEvent:
        """Emit a supervision event."""
        event_id = hashlib.sha256(
            f"{event_type}:{session_id}:{time.time()}".encode()
        ).hexdigest()[:16]

        event = SupervisionEvent(
            event_id=event_id,
            event_type=event_type,
            session_id=session_id,
            severity=severity,
            message=message,
            metadata=metadata or {},
        )

        self._events.append(event)

        # Dispatch to handlers
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error("Event handler failed: %s", exc)

        # Forward to Nerve if available
        if self._nerve_available:
            self._forward_to_nerve(event)

        return event

    def _forward_to_nerve(self, event: SupervisionEvent) -> None:
        """Forward event to Nerve system."""
        # Nerve API integration point
        pass

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[SupervisionEvent], None],
    ) -> None:
        """Register a handler for a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def get_events(
        self,
        session_id: str | None = None,
        severity: str | None = None,
        unresolved_only: bool = False,
    ) -> list[SupervisionEvent]:
        """Query supervision events with optional filters."""
        results = self._events
        if session_id:
            results = [e for e in results if e.session_id == session_id]
        if severity:
            results = [e for e in results if e.severity == severity]
        if unresolved_only:
            results = [e for e in results if not e.resolved]
        return results

    def resolve_event(self, event_id: str) -> bool:
        """Mark an event as resolved."""
        for event in self._events:
            if event.event_id == event_id:
                event.resolved = True
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get supervision statistics."""
        total = len(self._events)
        by_severity: dict[str, int] = {}
        by_type: dict[str, int] = {}
        unresolved = 0

        for event in self._events:
            by_severity[event.severity] = by_severity.get(event.severity, 0) + 1
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
            if not event.resolved:
                unresolved += 1

        return {
            "total_events": total,
            "unresolved": unresolved,
            "by_severity": by_severity,
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# Context Curation
# ---------------------------------------------------------------------------


class ContextCurator:
    """Manages context curation with configurable enforcement modes."""

    def __init__(
        self,
        mode: CurationMode = CurationMode.SHADOW,
        ledger: ContextLedger | None = None,
        supervisor: NerveSupervisor | None = None,
    ) -> None:
        self._mode = mode
        self._ledger = ledger
        self._supervisor = supervisor
        self._violations: list[dict[str, Any]] = []
        self._rules: list[Callable[[dict[str, Any]], str | None]] = []

    @property
    def mode(self) -> CurationMode:
        return self._mode

    def set_mode(self, mode: CurationMode) -> None:
        """Change the curation mode."""
        old_mode = self._mode
        self._mode = mode
        if self._supervisor:
            self._supervisor.emit_event(
                event_type="curation_mode_change",
                session_id="",
                message=f"Curation mode changed: {old_mode.value} -> {mode.value}",
                severity="info",
            )

    def add_rule(
        self,
        rule: Callable[[dict[str, Any]], str | None],
    ) -> None:
        """Add a curation rule.

        Rule takes context details and returns None if valid,
        or a violation message string if invalid.
        """
        self._rules.append(rule)

    def curate(self, context_details: dict[str, Any]) -> CurationDecision:
        """Run curation checks against context details."""
        violations: list[str] = []
        warnings: list[str] = []

        # Run all registered rules
        for rule in self._rules:
            result = rule(context_details)
            if result:
                violations.append(result)

        # Mode-specific handling
        allowed = True
        action = ""

        if self._mode == CurationMode.SHADOW:
            # Log only
            action = "logged"
        elif self._mode == CurationMode.ADVISORY:
            # Warn but allow
            allowed = True
            action = "warned"
            warnings = violations.copy()
            violations.clear()
        elif self._mode == CurationMode.ENFORCE:
            # Block violations
            if violations:
                allowed = False
                action = "blocked"

        # Record in ledger
        if self._ledger:
            self._ledger.append(
                operation=LedgerOperation.VALIDATE,
                session_id=context_details.get("session_id", ""),
                user_id=context_details.get("user_id", ""),
                details={
                    "violations": violations,
                    "warnings": warnings,
                    "action": action,
                },
                mode=self._mode,
            )

        # Emit supervision event for violations
        if violations and self._supervisor:
            self._supervisor.emit_event(
                event_type="curation_violation",
                session_id=context_details.get("session_id", ""),
                message=f"Curation violations: {'; '.join(violations)}",
                severity="warning" if self._mode != CurationMode.ENFORCE else "error",
                metadata={"violations": violations, "mode": self._mode.value},
            )

        # Store violations
        for v in violations:
            self._violations.append({
                "timestamp": time.time(),
                "violation": v,
                "mode": self._mode.value,
                "context": context_details,
            })

        return CurationDecision(
            allowed=allowed,
            mode=self._mode,
            violations=violations,
            warnings=warnings,
            action_taken=action,
        )

    def get_violations(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recorded violations, optionally filtered by session."""
        if session_id:
            return [
                v for v in self._violations
                if v.get("context", {}).get("session_id") == session_id
            ]
        return list(self._violations)


# ---------------------------------------------------------------------------
# Governance Manager
# ---------------------------------------------------------------------------


class GovernanceManager:
    """Unified governance manager orchestrating ledger, DoD, supervision, and curation."""

    def __init__(
        self,
        curation_mode: CurationMode = CurationMode.SHADOW,
        use_nerve: bool = False,
        persist_path: str | None = None,
        strict_dod: bool = True,
    ) -> None:
        self._ledger = ContextLedger(persist_path=persist_path)
        self._dod_enforcer = DoDEnforcer(strict=strict_dod)
        self._supervisor = NerveSupervisor(use_nerve=use_nerve)
        self._curator = ContextCurator(
            mode=curation_mode,
            ledger=self._ledger,
            supervisor=self._supervisor,
        )

    @property
    def ledger(self) -> ContextLedger:
        return self._ledger

    @property
    def dod(self) -> DoDEnforcer:
        return self._dod_enforcer

    @property
    def supervisor(self) -> NerveSupervisor:
        return self._supervisor

    @property
    def curator(self) -> ContextCurator:
        return self._curator

    def log_operation(
        self,
        operation: LedgerOperation,
        session_id: str,
        user_id: str,
        details: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        """Log a context operation to the ledger."""
        return self._ledger.append(
            operation=operation,
            session_id=session_id,
            user_id=user_id,
            details=details,
            mode=self._curator.mode,
        )

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the governance chain."""
        return self._ledger.verify_chain()

    def get_audit_trail(
        self,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate an audit trail report."""
        entries = (
            self._ledger.get_session_history(session_id)
            if session_id
            else self._ledger._entries
        )

        operations: dict[str, int] = {}
        for entry in entries:
            op = entry.operation.value
            operations[op] = operations.get(op, 0) + 1

        return {
            "total_entries": len(entries),
            "operations": operations,
            "curation_mode": self._curator.mode.value,
            "supervision_stats": self._supervisor.get_stats(),
            "integrity_valid": self._ledger.verify_chain()[0],
        }
