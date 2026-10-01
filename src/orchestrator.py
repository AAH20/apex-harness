"""Hierarchical task distribution engine for Apex Harness.

Implements Tier 0 (Director) → Tier 1 (Domain Orchestrators) →
Tier 2 (Squad Leaders) → Tier 3 (Workers) delegation with cost-aware
model routing and evidence-backed completion.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Tier(int, Enum):
    """Hierarchy tiers for task delegation."""

    DIRECTOR = 0
    DOMAIN_ORCHESTRATOR = 1
    SQUAD_LEADER = 2
    WORKER = 3


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class ModelTier(str, Enum):
    """Cost tiers for model routing."""

    FREE = "free"  # LongCat 2.5
    BUDGET = "budget"  # Codex
    VOLUME = "volume"  # Antigravity


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """A unit of work in the hierarchy."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    tier: Tier = Tier.WORKER
    domain: str = "general"
    difficulty: int = Field(default=1, ge=0, le=3)
    needs_tools: bool = False
    is_sensitive: bool = False
    requires_code: bool = False
    requires_volume: bool = False
    requires_orchestration: bool = False
    status: TaskStatus = TaskStatus.PENDING
    assigned_model: Optional[str] = None
    assigned_agent: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    result: Optional["TaskResult"] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: float = 300.0
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class TaskResult(BaseModel):
    """Evidence-backed result of task execution."""

    task_id: str
    success: bool
    output: Any = None
    evidence: list[str] = Field(default_factory=list)
    model_used: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None
    completed_at: float = Field(default_factory=time.time)
    verification_passed: bool = False
    verification_notes: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class AgentCapabilities(BaseModel):
    """Capabilities profile for an agent at a given tier."""

    agent_id: str
    tier: Tier
    model: str
    model_tier: ModelTier
    domains: list[str] = Field(default_factory=list)
    max_concurrent_tasks: int = 1
    current_load: int = 0
    is_available: bool = True
    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_tasks_completed: int = 0

    class Config:
        arbitrary_types_allowed = True


class OrchestratorConfig(BaseModel):
    """Configuration for the hierarchical orchestrator."""

    max_agents: int = 330
    max_concurrent_per_tier: dict[int, int] = Field(
        default_factory=lambda: {
            Tier.DIRECTOR: 1,
            Tier.DOMAIN_ORCHESTRATOR: 10,
            Tier.SQUAD_LEADER: 30,
            Tier.WORKER: 330,
        }
    )
    default_timeout_seconds: float = 300.0
    enable_evidence_verification: bool = True
    enable_cost_tracking: bool = True
    auto_escalate_on_failure: bool = True
    max_escalation_depth: int = 3

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class HierarchicalOrchestrator:
    """Hierarchical task distribution with cost-aware model routing.

    Manages the full Tier 0→1→2→3 hierarchy, assigns tasks to agents
    based on capabilities and cost, and enforces evidence-backed
    completion.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self._agents: dict[str, AgentCapabilities] = {}
        self._tasks: dict[str, Task] = {}
        self._task_queue: Optional[asyncio.PriorityQueue[tuple[int, str]]] = None
        self._running: bool = False
        self._lock: Optional[asyncio.Lock] = None
        self._model_router: Optional[Any] = None
        self._cost_tracker: Optional[Any] = None

        # Tier assignment counters for round-robin
        self._tier_counters: dict[Tier, int] = {t: 0 for t in Tier}

        logger.info(
            "HierarchicalOrchestrator initialized (max_agents=%d)",
            self.config.max_agents,
        )

    async def _get_lock(self) -> asyncio.Lock:
        """Get or lazily create the task lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _get_queue(self) -> asyncio.PriorityQueue[tuple[int, str]]:
        """Get or lazily create the task queue."""
        if self._task_queue is None:
            self._task_queue = asyncio.PriorityQueue()
        return self._task_queue

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, capabilities: AgentCapabilities) -> None:
        """Register an agent with the orchestrator.

        Args:
            capabilities: The agent's capability profile.

        Raises:
            ValueError: If agent_id already registered or max_agents exceeded.
        """
        if capabilities.agent_id in self._agents:
            raise ValueError(f"Agent '{capabilities.agent_id}' already registered")
        if len(self._agents) >= self.config.max_agents:
            raise ValueError(
                f"Max agents ({self.config.max_agents}) reached"
            )
        self._agents[capabilities.agent_id] = capabilities
        logger.info(
            "Registered agent %s (tier=%s, model=%s)",
            capabilities.agent_id,
            capabilities.tier.name,
            capabilities.model,
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the orchestrator."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info("Unregistered agent %s", agent_id)

    def set_model_router(self, router: Any) -> None:
        """Set the model router for cost-aware routing decisions."""
        self._model_router = router

    def set_cost_tracker(self, tracker: Any) -> None:
        """Set the cost tracker for budget enforcement."""
        self._cost_tracker = tracker

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    async def submit_task(self, task: Task) -> str:
        """Submit a task to the orchestration queue.

        Args:
            task: The task to execute.

        Returns:
            The task ID.
        """
        lock = await self._get_lock()
        async with lock:
            self._tasks[task.id] = task

        # Priority: lower difficulty = higher priority (executed first)
        # Tier 0 tasks get highest priority
        priority = task.tier.value * 10 + task.difficulty
        queue = await self._get_queue()
        await queue.put((priority, task.id))

        logger.info(
            "Submitted task %s (tier=%s, domain=%s, difficulty=%d)",
            task.id,
            task.tier.name,
            task.domain,
            task.difficulty,
        )
        return task.id

    async def submit_tasks(self, tasks: list[Task]) -> list[str]:
        """Submit multiple tasks at once."""
        ids: list[str] = []
        for task in tasks:
            task_id = await self.submit_task(task)
            ids.append(task_id)
        return ids

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        return self._tasks.get(task_id)

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Retrieve the result of a completed task."""
        task = self._tasks.get(task_id)
        return task.result if task else None

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose(
        self,
        goal: str,
        domains: Optional[list[str]] = None,
        max_subtasks: int = 10,
    ) -> list[Task]:
        """Decompose a high-level goal into domain-specific subtasks.

        In production this would use an LLM to decompose. Here we provide
        a deterministic decomposition based on domain keywords.

        Args:
            goal: The high-level goal description.
            domains: Optional list of domains to decompose into.
            max_subtasks: Maximum number of subtasks to create.

        Returns:
            List of subtask Task objects at Tier 1.
        """
        if domains is None:
            domains = self._infer_domains(goal)

        subtasks: list[Task] = []
        for i, domain in enumerate(domains[:max_subtasks]):
            task = Task(
                description=f"[{domain}] {goal}",
                tier=Tier.DOMAIN_ORCHESTRATOR,
                domain=domain,
                difficulty=2,
                requires_orchestration=True,
                acceptance_criteria=[
                    f"Domain '{domain}' subtasks defined",
                    f"Agents assigned for '{domain}'",
                ],
            )
            subtasks.append(task)

        logger.info(
            "Decomposed goal into %d domain subtasks: %s",
            len(subtasks),
            [t.domain for t in subtasks],
        )
        return subtasks

    def _infer_domains(self, goal: str) -> list[str]:
        """Infer relevant domains from a goal description."""
        domain_keywords: dict[str, list[str]] = {
            "infrastructure": ["infra", "deploy", "docker", "k8s", "ci/cd"],
            "code": ["code", "implement", "build", "develop", "refactor"],
            "research": ["research", "analyze", "investigate", "survey"],
            "data": ["data", "pipeline", "etl", "analytics", "ml"],
            "security": ["security", "audit", "vulnerability", "penetration"],
            "documentation": ["doc", "write", "guide", "readme", "api"],
            "testing": ["test", "qa", "coverage", "e2e", "integration"],
        }

        goal_lower = goal.lower()
        matched: list[str] = []
        for domain, keywords in domain_keywords.items():
            if any(kw in goal_lower for kw in keywords):
                matched.append(domain)

        return matched if matched else ["general"]

    # ------------------------------------------------------------------
    # Model routing
    # ------------------------------------------------------------------

    def route_task(self, task: Task) -> str:
        """Determine the best model for a task based on its properties.

        Uses the model router if available, otherwise falls back to
        deterministic routing rules.

        Args:
            task: The task to route.

        Returns:
            The selected model name.
        """
        if self._model_router is not None:
            try:
                decision = self._model_router.route(task)
                return decision.selected_model
            except Exception as exc:
                logger.warning("Model router failed (%s); using fallback", exc)

        return self._fallback_route(task)

    def _fallback_route(self, task: Task) -> str:
        """Deterministic fallback routing when no router is available.

        Routing rules (from hierarchical-orchestrator skill):
        - requires_code + budget available → Codex
        - requires_volume → Antigravity Flash
        - requires_orchestration → LongCat 2.5
        - default → LongCat 2.5
        """
        if task.requires_code:
            return "codex"
        if task.requires_volume:
            return "antigravity-flash"
        if task.requires_orchestration:
            return "longcat-2.5"
        return "longcat-2.5"

    # ------------------------------------------------------------------
    # Agent assignment
    # ------------------------------------------------------------------

    def assign_agent(self, task: Task) -> Optional[str]:
        """Assign the best available agent for a task.

        Selection criteria (in priority order):
        1. Tier match
        2. Domain match
        3. Availability (current_load < max_concurrent)
        4. Lowest cost
        5. Highest success rate

        Args:
            task: The task to assign.

        Returns:
            The selected agent ID, or None if no agent available.
        """
        candidates: list[AgentCapabilities] = []

        for agent in self._agents.values():
            if agent.tier != task.tier:
                continue
            if not agent.is_available:
                continue
            if agent.current_load >= agent.max_concurrent_tasks:
                continue
            candidates.append(agent)

        if not candidates:
            logger.warning(
                "No available agents for tier %s (task %s)",
                task.tier.name,
                task.id,
            )
            return None

        # Score: domain_match * 2 + (1 - normalized_cost) + success_rate
        def score(agent: AgentCapabilities) -> float:
            domain_match = 1.0 if task.domain in agent.domains else 0.0
            max_cost = max((a.cost_per_1k_tokens for a in candidates), default=1.0)
            cost_score = 1.0 - (agent.cost_per_1k_tokens / max_cost) if max_cost > 0 else 1.0
            return domain_match * 2.0 + cost_score + agent.success_rate

        best = max(candidates, key=score)
        best.current_load += 1
        task.assigned_agent = best.agent_id
        task.assigned_model = best.model
        task.status = TaskStatus.ASSIGNED

        logger.info(
            "Assigned task %s to agent %s (model=%s, tier=%s)",
            task.id,
            best.agent_id,
            best.model,
            best.tier.name,
        )
        return best.agent_id

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_task(
        self,
        task: Task,
        handler: Optional[Callable[[Task], Coroutine[Any, Any, TaskResult]]] = None,
    ) -> TaskResult:
        """Execute a single task with the assigned agent.

        Args:
            task: The task to execute.
            handler: Optional async handler that performs the actual work.
                     If None, a simulated result is produced.

        Returns:
            The task result with evidence.
        """
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()

        if task.assigned_agent is None:
            self.assign_agent(task)

        if task.assigned_agent is None:
            task.status = TaskStatus.FAILED
            result = TaskResult(
                task_id=task.id,
                success=False,
                error="No available agent for task",
            )
            task.result = result
            return result

        try:
            if handler is not None:
                result = await asyncio.wait_for(
                    handler(task), timeout=task.timeout_seconds
                )
            else:
                result = await self._simulate_execution(task)

            # Verify evidence
            if self.config.enable_evidence_verification:
                result = self._verify_result(task, result)

            task.result = result
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.completed_at = time.time()

            # Update agent stats
            agent = self._agents.get(task.assigned_agent)
            if agent:
                agent.current_load = max(0, agent.current_load - 1)
                agent.total_tasks_completed += 1
                if not result.success:
                    agent.success_rate = max(0.0, agent.success_rate - 0.05)

            logger.info(
                "Task %s %s (model=%s, cost=$%.4f, tokens=%d)",
                task.id,
                "completed" if result.success else "FAILED",
                result.model_used or "unknown",
                result.cost_usd,
                result.tokens_used,
            )

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            result = TaskResult(
                task_id=task.id,
                success=False,
                error=f"Timeout after {task.timeout_seconds}s",
            )
            task.result = result
            task.completed_at = time.time()
            logger.error("Task %s timed out", task.id)

        except Exception as exc:
            task.status = TaskStatus.FAILED
            result = TaskResult(
                task_id=task.id,
                success=False,
                error=str(exc),
            )
            task.result = result
            task.completed_at = time.time()
            logger.exception("Task %s failed: %s", task.id, exc)

        # Auto-escalate on failure
        if (
            not result.success
            and self.config.auto_escalate_on_failure
            and task.retry_count < task.max_retries
        ):
            await self._escalate_task(task, result)

        return result

    async def _simulate_execution(self, task: Task) -> TaskResult:
        """Simulate task execution (placeholder for real agent dispatch)."""
        await asyncio.sleep(0.01)  # Minimal delay for async semantics
        return TaskResult(
            task_id=task.id,
            success=True,
            output=f"Completed: {task.description}",
            evidence=[
                f"Task '{task.description}' executed successfully",
                f"Model: {task.assigned_model}",
                f"Agent: {task.assigned_agent}",
            ],
            model_used=task.assigned_model,
            tokens_used=150,
            cost_usd=0.0,
            latency_ms=10.0,
        )

    def _verify_result(self, task: Task, result: TaskResult) -> TaskResult:
        """Verify that a task result has sufficient evidence.

        Args:
            task: The original task.
            result: The result to verify.

        Returns:
            The result with verification metadata attached.
        """
        if not result.success:
            result.verification_passed = False
            result.verification_notes = "Task failed; no verification needed"
            return result

        # Check evidence presence
        has_evidence = len(result.evidence) > 0
        has_output = result.output is not None

        # Check acceptance criteria
        criteria_met = all(
            any(criterion.lower() in ev.lower() for ev in result.evidence)
            for criterion in task.acceptance_criteria
        ) if task.acceptance_criteria else True

        result.verification_passed = has_evidence and has_output and criteria_met
        result.verification_notes = (
            f"evidence={has_evidence}, output={has_output}, "
            f"criteria_met={criteria_met}"
        )

        if not result.verification_passed:
            logger.warning(
                "Task %s verification failed: %s",
                task.id,
                result.verification_notes,
            )

        return result

    async def _escalate_task(self, task: Task, failed_result: TaskResult) -> None:
        """Escalate a failed task to a higher tier or retry."""
        task.retry_count += 1

        if task.tier > Tier.DIRECTOR:
            # Escalate to parent tier
            new_tier = Tier(task.tier.value - 1)
            logger.warning(
                "Escalating task %s from %s to %s (retry %d/%d)",
                task.id,
                task.tier.name,
                new_tier.name,
                task.retry_count,
                task.max_retries,
            )
            task.tier = new_tier
            task.status = TaskStatus.ESCALATED
            task.assigned_agent = None
            task.assigned_model = None
            task.result = None
            task.started_at = None
            task.completed_at = None

            # Re-submit with higher priority
            priority = new_tier.value * 10 + task.difficulty - 1
            queue = await self._get_queue()
            await queue.put((priority, task.id))
        else:
            # Already at Director tier — just retry
            logger.warning(
                "Retrying task %s at Director tier (retry %d/%d)",
                task.id,
                task.retry_count,
                task.max_retries,
            )
            task.status = TaskStatus.PENDING
            task.assigned_agent = None
            task.assigned_model = None
            task.result = None
            task.started_at = None
            task.completed_at = None
            queue = await self._get_queue()
            await queue.put((task.tier.value * 10 + task.difficulty, task.id))

    # ------------------------------------------------------------------
    # Hierarchical dispatch
    # ------------------------------------------------------------------

    async def dispatch_hierarchy(
        self,
        goal: str,
        handler: Optional[Callable[[Task], Coroutine[Any, Any, TaskResult]]] = None,
    ) -> list[TaskResult]:
        """Execute a full hierarchical dispatch for a goal.

        Flow:
        1. Decompose goal into Tier 1 domain subtasks
        2. For each Tier 1 task, decompose into Tier 2 squad tasks
        3. For each Tier 2 task, decompose into Tier 3 worker tasks
        4. Execute all Tier 3 tasks in parallel
        5. Aggregate results back up the hierarchy

        Args:
            goal: The high-level goal.
            handler: Optional task execution handler.

        Returns:
            List of all task results.
        """
        # Step 1: Decompose into Tier 1
        tier1_tasks = self.decompose(goal)
        tier1_ids = await self.submit_tasks(tier1_tasks)

        # Step 2: Decompose Tier 1 → Tier 2
        tier2_tasks: list[Task] = []
        for t1 in tier1_tasks:
            subtasks = self.decompose(
                t1.description,
                domains=[t1.domain],
                max_subtasks=3,
            )
            for st in subtasks:
                st.tier = Tier.SQUAD_LEADER
                st.parent_id = t1.id
                st.difficulty = max(1, t1.difficulty - 1)
                t1.children_ids.append(st.id)
            tier2_tasks.extend(subtasks)

        tier2_ids = await self.submit_tasks(tier2_tasks)

        # Step 3: Decompose Tier 2 → Tier 3
        tier3_tasks: list[Task] = []
        for t2 in tier2_tasks:
            subtasks = self.decompose(
                t2.description,
                domains=[t2.domain],
                max_subtasks=3,
            )
            for st in subtasks:
                st.tier = Tier.WORKER
                st.parent_id = t2.id
                st.difficulty = max(1, t2.difficulty - 1)
                st.requires_code = "code" in st.domain
                st.requires_volume = "data" in st.domain
                t2.children_ids.append(st.id)
            tier3_tasks.extend(subtasks)

        tier3_ids = await self.submit_tasks(tier3_tasks)

        # Step 4: Execute all tasks (Tier 3 first, then aggregate up)
        all_results: list[TaskResult] = []

        # Execute Tier 3 in parallel
        tier3_futures = [
            self.execute_task(self._tasks[tid], handler) for tid in tier3_ids
        ]
        tier3_results = await asyncio.gather(*tier3_futures, return_exceptions=True)
        for res in tier3_results:
            if isinstance(res, TaskResult):
                all_results.append(res)

        # Execute Tier 2
        tier2_futures = [
            self.execute_task(self._tasks[tid], handler) for tid in tier2_ids
        ]
        tier2_results = await asyncio.gather(*tier2_futures, return_exceptions=True)
        for res in tier2_results:
            if isinstance(res, TaskResult):
                all_results.append(res)

        # Execute Tier 1
        tier1_futures = [
            self.execute_task(self._tasks[tid], handler) for tid in tier1_ids
        ]
        tier1_results = await asyncio.gather(*tier1_futures, return_exceptions=True)
        for res in tier1_results:
            if isinstance(res, TaskResult):
                all_results.append(res)

        logger.info(
            "Hierarchical dispatch complete: %d tasks executed, %d succeeded",
            len(all_results),
            sum(1 for r in all_results if r.success),
        )
        return all_results

    # ------------------------------------------------------------------
    # Status & monitoring
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current orchestrator status."""
        tier_counts: dict[str, int] = {t.name: 0 for t in Tier}
        for agent in self._agents.values():
            tier_counts[agent.tier.name] += 1

        task_status_counts: dict[str, int] = {s.value: 0 for s in TaskStatus}
        for task in self._tasks.values():
            task_status_counts[task.status.value] += 1

        return {
            "total_agents": len(self._agents),
            "tier_distribution": tier_counts,
            "total_tasks": len(self._tasks),
            "task_status": task_status_counts,
            "queue_size": self._task_queue.qsize() if self._task_queue else 0,
            "running": self._running,
        }

    async def start(self) -> None:
        """Start the orchestrator's background processing loop."""
        self._running = True
        logger.info("Orchestrator started")

        while self._running:
            try:
                queue = await self._get_queue()
                priority, task_id = await asyncio.wait_for(
                    queue.get(), timeout=1.0
                )
                task = self._tasks.get(task_id)
                if task and task.status == TaskStatus.PENDING:
                    asyncio.create_task(self.execute_task(task))
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.exception("Orchestrator loop error: %s", exc)

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        logger.info("Orchestrator stopped")
