"""Agent slot management and wave-based dispatch for Apex Harness.

Manages 330 agent slots, wave-based parallel dispatch, and concurrency
control across the hierarchical orchestration stack.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SlotStatus(str, Enum):
    """Status of an agent slot."""

    AVAILABLE = "available"
    RESERVED = "reserved"
    BUSY = "busy"
    DRAINING = "draining"
    DISABLED = "disabled"


class WaveStatus(str, Enum):
    """Status of a dispatch wave."""

    PENDING = "pending"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class AgentSlot(BaseModel):
    """A single agent slot in the scheduler."""

    slot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    tier: int = Field(default=3, ge=0, le=3)
    status: SlotStatus = SlotStatus.AVAILABLE
    current_task_id: Optional[str] = None
    model: Optional[str] = None
    assigned_at: Optional[float] = None
    released_at: Optional[float] = None
    total_tasks_completed: int = 0
    total_busy_time_ms: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class Wave(BaseModel):
    """A dispatch wave containing multiple agent slots."""

    wave_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    slot_ids: list[str] = Field(default_factory=list)
    status: WaveStatus = WaveStatus.PENDING
    max_concurrency: int = 50
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class WaveResult(BaseModel):
    """Result of a completed wave."""

    wave_id: str
    total_tasks: int
    succeeded: int
    failed: int
    total_latency_ms: float
    avg_latency_ms: float
    cost_usd: float
    completed_at: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True


class SchedulerConfig(BaseModel):
    """Configuration for the scheduler."""

    total_slots: int = 330
    default_wave_size: int = 50
    max_concurrent_waves: int = 3
    slot_timeout_seconds: float = 300.0
    enable_auto_scaling: bool = True
    drain_timeout_seconds: float = 60.0

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """330-agent slot scheduler with wave-based dispatch.

    Manages a pool of agent slots, organizes them into waves for
    parallel execution, and enforces concurrency limits.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self.config = config or SchedulerConfig()
        self._slots: dict[str, AgentSlot] = {}
        self._waves: dict[str, Wave] = {}
        self._wave_results: dict[str, WaveResult] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._slot_lock: Optional[asyncio.Lock] = None
        self._running = False

        # Initialize slots
        self._initialize_slots()

        logger.info(
            "Scheduler initialized (%d slots, max_concurrent_waves=%d)",
            self.config.total_slots,
            self.config.max_concurrent_waves,
        )

    def _initialize_slots(self) -> None:
        """Create the initial pool of agent slots."""
        for i in range(self.config.total_slots):
            # Distribute tiers: 1 Director, 10 Orchestrators, 30 Squad Leaders, rest Workers
            if i == 0:
                tier = 0
            elif i <= 10:
                tier = 1
            elif i <= 40:
                tier = 2
            else:
                tier = 3

            slot = AgentSlot(tier=tier)
            self._slots[slot.slot_id] = slot

        tier_counts: dict[int, int] = {}
        for slot in self._slots.values():
            tier_counts[slot.tier] = tier_counts.get(slot.tier, 0) + 1

        logger.info("Slot tier distribution: %s", tier_counts)

    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or lazily create the wave-level semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_waves)
        return self._semaphore

    async def _get_lock(self) -> asyncio.Lock:
        """Get or lazily create the slot lock."""
        if self._slot_lock is None:
            self._slot_lock = asyncio.Lock()
        return self._slot_lock

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    async def acquire_slot(
        self,
        tier: int = 3,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[AgentSlot]:
        """Acquire an available slot for task execution.

        Args:
            tier: The hierarchy tier needed.
            agent_id: Optional specific agent ID.
            model: Optional model to assign.

        Returns:
            The acquired slot, or None if no slot available.
        """
        lock = await self._get_lock()
        async with lock:
            # Find available slot matching tier
            for slot in self._slots.values():
                if slot.tier == tier and slot.status == SlotStatus.AVAILABLE:
                    slot.status = SlotStatus.BUSY
                    slot.agent_id = agent_id
                    slot.model = model
                    slot.current_task_id = None
                    slot.assigned_at = time.time()
                    slot.released_at = None
                    logger.debug(
                        "Acquired slot %s (tier=%d, agent=%s)",
                        slot.slot_id,
                        tier,
                        agent_id,
                    )
                    return slot

            logger.warning("No available slot for tier %d", tier)
            return None

    async def release_slot(
        self,
        slot_id: str,
        task_id: Optional[str] = None,
    ) -> None:
        """Release a slot back to the available pool.

        Args:
            slot_id: The slot to release.
            task_id: Optional task ID that was running on this slot.
        """
        lock = await self._get_lock()
        async with lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                logger.warning("Cannot release unknown slot %s", slot_id)
                return

            if slot.assigned_at:
                busy_time = (time.time() - slot.assigned_at) * 1000
                slot.total_busy_time_ms += busy_time

            if task_id:
                slot.total_tasks_completed += 1

            slot.status = SlotStatus.AVAILABLE
            slot.agent_id = None
            slot.model = None
            slot.current_task_id = None
            slot.assigned_at = None
            slot.released_at = time.time()

            logger.debug("Released slot %s", slot_id)

    async def get_slot(self, slot_id: str) -> Optional[AgentSlot]:
        """Get a slot by ID."""
        return self._slots.get(slot_id)

    async def get_available_slots(self, tier: Optional[int] = None) -> list[AgentSlot]:
        """Get all available slots, optionally filtered by tier."""
        slots = self._slots.values()
        if tier is not None:
            slots = [s for s in slots if s.tier == tier]
        return [s for s in slots if s.status == SlotStatus.AVAILABLE]

    async def disable_slot(self, slot_id: str) -> None:
        """Disable a slot (for maintenance or error recovery)."""
        lock = await self._get_lock()
        async with lock:
            slot = self._slots.get(slot_id)
            if slot:
                slot.status = SlotStatus.DISABLED
                logger.info("Disabled slot %s", slot_id)

    async def enable_slot(self, slot_id: str) -> None:
        """Re-enable a disabled slot."""
        lock = await self._get_lock()
        async with lock:
            slot = self._slots.get(slot_id)
            if slot and slot.status == SlotStatus.DISABLED:
                slot.status = SlotStatus.AVAILABLE
                logger.info("Enabled slot %s", slot_id)

    # ------------------------------------------------------------------
    # Wave management
    # ------------------------------------------------------------------

    async def create_wave(
        self,
        name: str,
        size: Optional[int] = None,
        tier: int = 3,
        max_concurrency: Optional[int] = None,
    ) -> Wave:
        """Create a new dispatch wave.

        Args:
            name: Human-readable wave name.
            size: Number of slots to reserve (default: config.default_wave_size).
            tier: Tier for this wave.
            max_concurrency: Max concurrent tasks within the wave.

        Returns:
            The created wave.

        Raises:
            ValueError: If not enough slots available.
        """
        size = size or self.config.default_wave_size
        max_concurrency = max_concurrency or min(size, 50)

        lock = await self._get_lock()
        async with lock:
            available = [
                s for s in self._slots.values()
                if s.tier == tier and s.status == SlotStatus.AVAILABLE
            ]

            if len(available) < size:
                raise ValueError(
                    f"Not enough available slots for tier {tier}: "
                    f"need {size}, have {len(available)}"
                )

            # Reserve slots
            reserved = available[:size]
            slot_ids: list[str] = []
            for slot in reserved:
                slot.status = SlotStatus.RESERVED
                slot_ids.append(slot.slot_id)

            wave = Wave(
                name=name,
                slot_ids=slot_ids,
                max_concurrency=max_concurrency,
                metadata={"tier": tier, "size": size},
            )
            self._waves[wave.wave_id] = wave

            logger.info(
                "Created wave '%s' (%s) with %d slots (tier=%d)",
                name,
                wave.wave_id,
                size,
                tier,
            )
            return wave

    async def dispatch_wave(
        self,
        wave_id: str,
        task_handler: Callable[[AgentSlot], Coroutine[Any, Any, Any]],
    ) -> WaveResult:
        """Dispatch a wave of tasks across reserved slots.

        Args:
            wave_id: The wave to dispatch.
            task_handler: Async handler that processes a single slot.

        Returns:
            The wave result.

        Raises:
            ValueError: If wave not found.
        """
        wave = self._waves.get(wave_id)
        if wave is None:
            raise ValueError(f"Wave '{wave_id}' not found")

        wave.status = WaveStatus.DISPATCHING
        wave.started_at = time.time()

        logger.info(
            "Dispatching wave '%s' (%d slots, max_concurrency=%d)",
            wave.name,
            len(wave.slot_ids),
            wave.max_concurrency,
        )

        # Use semaphore to limit concurrency within the wave
        wave_semaphore = asyncio.Semaphore(wave.max_concurrency)

        async def _run_slot(slot_id: str) -> dict[str, Any]:
            async with wave_semaphore:
                slot = self._slots.get(slot_id)
                if slot is None:
                    return {"success": False, "error": "Slot not found"}

                slot.status = SlotStatus.BUSY
                slot.assigned_at = time.time()

                try:
                    start = time.monotonic()
                    output = await asyncio.wait_for(
                        task_handler(slot),
                        timeout=self.config.slot_timeout_seconds,
                    )
                    latency = (time.monotonic() - start) * 1000

                    return {
                        "success": True,
                        "output": output,
                        "latency_ms": latency,
                        "slot_id": slot_id,
                    }
                except asyncio.TimeoutError:
                    return {
                        "success": False,
                        "error": "Timeout",
                        "latency_ms": self.config.slot_timeout_seconds * 1000,
                        "slot_id": slot_id,
                    }
                except Exception as exc:
                    return {
                        "success": False,
                        "error": str(exc),
                        "latency_ms": 0.0,
                        "slot_id": slot_id,
                    }
                finally:
                    slot.status = SlotStatus.AVAILABLE
                    slot.total_tasks_completed += 1
                    if slot.assigned_at:
                        slot.total_busy_time_ms += (
                            time.time() - slot.assigned_at
                        ) * 1000
                    slot.assigned_at = None
                    slot.current_task_id = None

        # Execute all slots in the wave
        sem = await self._get_semaphore()
        async with sem:
            wave.status = WaveStatus.RUNNING
            results = await asyncio.gather(
                *[_run_slot(sid) for sid in wave.slot_ids],
                return_exceptions=True,
            )

        # Aggregate results
        succeeded = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - succeeded
        latencies = [
            r.get("latency_ms", 0.0)
            for r in results
            if isinstance(r, dict)
        ]
        total_latency = sum(latencies)
        avg_latency = total_latency / len(latencies) if latencies else 0.0

        wave.status = WaveStatus.COMPLETED
        wave.completed_at = time.time()

        result = WaveResult(
            wave_id=wave_id,
            total_tasks=len(results),
            succeeded=succeeded,
            failed=failed,
            total_latency_ms=total_latency,
            avg_latency_ms=avg_latency,
            cost_usd=0.0,  # Populated by cost tracker integration
        )
        self._wave_results[wave_id] = result

        logger.info(
            "Wave '%s' completed: %d/%d succeeded, avg_latency=%.1fms",
            wave.name,
            succeeded,
            len(results),
            avg_latency,
        )
        return result

    async def cancel_wave(self, wave_id: str) -> None:
        """Cancel a running wave and release its slots."""
        wave = self._waves.get(wave_id)
        if wave is None:
            return

        wave.status = WaveStatus.CANCELLED

        # Release all slots
        for slot_id in wave.slot_ids:
            await self.release_slot(slot_id)

        logger.info("Cancelled wave '%s'", wave.name)

    # ------------------------------------------------------------------
    # Multi-wave dispatch
    # ------------------------------------------------------------------

    async def dispatch_waves(
        self,
        waves_config: list[dict[str, Any]],
        task_handler: Callable[[AgentSlot], Coroutine[Any, Any, Any]],
    ) -> list[WaveResult]:
        """Dispatch multiple waves sequentially or in parallel.

        Args:
            waves_config: List of wave configs, each with 'name', 'size',
                         'tier', 'max_concurrency'.
            task_handler: Task handler for each slot.

        Returns:
            List of wave results.
        """
        # Create all waves first
        created_waves: list[Wave] = []
        for wc in waves_config:
            try:
                wave = await self.create_wave(
                    name=wc.get("name", f"wave-{len(created_waves)}"),
                    size=wc.get("size"),
                    tier=wc.get("tier", 3),
                    max_concurrency=wc.get("max_concurrency"),
                )
                created_waves.append(wave)
            except ValueError as exc:
                logger.error("Failed to create wave: %s", exc)

        # Dispatch waves (respecting max_concurrent_waves via semaphore)
        results: list[WaveResult] = []
        for wave in created_waves:
            result = await self.dispatch_wave(wave.wave_id, task_handler)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Status & monitoring
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current scheduler status."""
        slot_status_counts: dict[str, int] = {}
        for slot in self._slots.values():
            slot_status_counts[slot.status.value] = (
                slot_status_counts.get(slot.status.value, 0) + 1
            )

        tier_distribution: dict[int, dict[str, int]] = {}
        for slot in self._slots.values():
            tier_key = f"tier_{slot.tier}"
            if tier_key not in tier_distribution:
                tier_distribution[tier_key] = {}
            status = slot.status.value
            tier_distribution[tier_key][status] = (
                tier_distribution[tier_key].get(status, 0) + 1
            )

        return {
            "total_slots": len(self._slots),
            "slot_status": slot_status_counts,
            "tier_distribution": tier_distribution,
            "total_waves": len(self._waves),
            "active_waves": sum(
                1 for w in self._waves.values()
                if w.status in (WaveStatus.DISPATCHING, WaveStatus.RUNNING)
            ),
            "running": self._running,
        }

    async def drain_all(self, timeout_seconds: Optional[float] = None) -> None:
        """Drain all active waves and wait for completion.

        Args:
            timeout_seconds: Max time to wait for draining.
        """
        timeout = timeout_seconds or self.config.drain_timeout_seconds
        logger.info("Draining all active waves (timeout=%.0fs)", timeout)

        active_waves = [
            w for w in self._waves.values()
            if w.status in (WaveStatus.DISPATCHING, WaveStatus.RUNNING)
        ]

        for wave in active_waves:
            wave.status = SlotStatus.DRAINING

        # Wait for all slots to become available
        start = time.time()
        while time.time() - start < timeout:
            busy = [
                s for s in self._slots.values()
                if s.status == SlotStatus.BUSY
            ]
            if not busy:
                break
            await asyncio.sleep(0.5)

        logger.info("All waves drained")

    async def start(self) -> None:
        """Start the scheduler background loop."""
        self._running = True
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler and drain active work."""
        self._running = False
        await self.drain_all()
        logger.info("Scheduler stopped")
