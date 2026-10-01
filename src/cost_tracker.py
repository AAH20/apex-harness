"""Per-call cost tracking and budget enforcement for Apex Harness.

Tracks token usage, enforces budgets per model, and provides
cost visibility across the orchestration stack.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BudgetPeriod(str, Enum):
    """Budget enforcement period."""

    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    TOTAL = "total"


class AlertSeverity(str, Enum):
    """Cost alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token usage for a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""
    timestamp: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True


class CostRecord(BaseModel):
    """A single cost record."""

    record_id: str = Field(default_factory=lambda: f"cr-{int(time.time() * 1000000)}")
    model: str
    tokens: TokenUsage
    cost_usd: float
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    tier: int = 3
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class BudgetConfig(BaseModel):
    """Budget configuration for a model."""

    model: str
    period: BudgetPeriod
    limit_usd: float
    alert_threshold_pct: float = 0.8  # Alert at 80% of budget
    hard_stop: bool = True  # Reject new calls when budget exceeded

    class Config:
        arbitrary_types_allowed = True


class CostAlert(BaseModel):
    """A cost budget alert."""

    alert_id: str = Field(default_factory=lambda: f"alert-{int(time.time() * 1000000)}")
    model: str
    period: BudgetPeriod
    current_spend: float
    budget_limit: float
    threshold_pct: float
    severity: AlertSeverity
    message: str
    timestamp: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True


class CostSummary(BaseModel):
    """Aggregated cost summary."""

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_calls: int = 0
    by_model: dict[str, float] = Field(default_factory=dict)
    by_tier: dict[int, float] = Field(default_factory=dict)
    by_period: dict[str, float] = Field(default_factory=dict)
    avg_cost_per_call: float = 0.0
    avg_tokens_per_call: float = 0.0

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when a budget limit would be exceeded."""

    def __init__(
        self,
        model: str,
        period: BudgetPeriod,
        current_spend: float,
        limit: float,
        attempted_cost: float,
    ) -> None:
        self.model = model
        self.period = period
        self.current_spend = current_spend
        self.limit = limit
        self.attempted_cost = attempted_cost
        super().__init__(
            f"Budget exceeded for {model} ({period.value}): "
            f"spend=${current_spend:.4f}, limit=${limit:.4f}, "
            f"attempted=${attempted_cost:.4f}"
        )


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------


# Model pricing (per 1M tokens, USD)
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "longcat-2.5": {"prompt": 0.0, "completion": 0.0, "cached": 0.0},
    "codex": {"prompt": 1.25, "completion": 10.0, "cached": 0.125},
    "antigravity-flash": {"prompt": 0.30, "completion": 2.50, "cached": 0.03},
    "gpt-4o": {"prompt": 2.50, "completion": 10.0, "cached": 1.25},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60, "cached": 0.075},
    "claude-sonnet-4": {"prompt": 3.0, "completion": 15.0, "cached": 0.30},
    "claude-haiku-4": {"prompt": 0.80, "completion": 4.0, "cached": 0.08},
}


class CostTracker:
    """Per-call cost tracking with budget enforcement.

    Tracks every API call's token usage and cost, enforces
    configurable budgets per model, and emits alerts at thresholds.
    """

    def __init__(
        self,
        model_pricing: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self._pricing = model_pricing or DEFAULT_MODEL_PRICING
        self._records: list[CostRecord] = []
        self._budgets: dict[str, list[BudgetConfig]] = defaultdict(list)
        self._alerts: list[CostAlert] = []
        self._lock: Optional[asyncio.Lock] = None
        self._alert_handlers: list[Callable[[CostAlert], Any]] = []

        # Running totals
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._total_calls: int = 0
        self._cost_by_model: dict[str, float] = defaultdict(float)
        self._cost_by_tier: dict[int, float] = defaultdict(float)
        self._tokens_by_model: dict[str, int] = defaultdict(int)

        logger.info("CostTracker initialized")

    async def _get_lock(self) -> asyncio.Lock:
        """Get or lazily create the record lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # Budget configuration
    # ------------------------------------------------------------------

    def set_budget(self, config: BudgetConfig) -> None:
        """Set a budget for a model.

        Args:
            config: The budget configuration.
        """
        model_budgets = self._budgets[config.model]
        # Replace existing budget for same period
        model_budgets[:] = [
            b for b in model_budgets if b.period != config.period
        ]
        model_budgets.append(config)
        logger.info(
            "Budget set: %s %s = $%.2f (alert at %.0f%%)",
            config.model,
            config.period.value,
            config.limit_usd,
            config.alert_threshold_pct * 100,
        )

    def set_default_budgets(self) -> None:
        """Set default budgets matching the Apex Harness cost model.

        - LongCat 2.5: Free, unlimited
        - Codex: $20/month (20% of $100/mo plan)
        - Antigravity: Subscription, unlimited
        """
        self.set_budget(
            BudgetConfig(
                model="codex",
                period=BudgetPeriod.MONTHLY,
                limit_usd=20.0,
                alert_threshold_pct=0.8,
                hard_stop=True,
            )
        )
        self.set_budget(
            BudgetConfig(
                model="codex",
                period=BudgetPeriod.DAILY,
                limit_usd=1.0,
                alert_threshold_pct=0.8,
                hard_stop=True,
            )
        )
        logger.info("Default budgets configured")

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def compute_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Compute the cost of an API call.

        Args:
            model: The model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            cached_tokens: Number of cached tokens (discounted).

        Returns:
            Cost in USD.
        """
        pricing = self._pricing.get(model)
        if pricing is None:
            logger.warning("Unknown model '%s'; cost set to $0", model)
            return 0.0

        effective_prompt = prompt_tokens - cached_tokens
        cost = (
            effective_prompt * pricing["prompt"]
            + completion_tokens * pricing["completion"]
            + cached_tokens * pricing["cached"]
        ) / 1_000_000

        return cost

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tier: int = 3,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CostRecord:
        """Record an API call and check budgets.

        Args:
            model: The model used.
            prompt_tokens: Prompt tokens consumed.
            completion_tokens: Completion tokens generated.
            cached_tokens: Cached tokens (discounted).
            task_id: Optional associated task ID.
            agent_id: Optional associated agent ID.
            tier: Hierarchy tier.
            metadata: Optional additional metadata.

        Returns:
            The cost record.

        Raises:
            BudgetExceededError: If a hard-stop budget would be exceeded.
        """
        cost = self.compute_cost(model, prompt_tokens, completion_tokens, cached_tokens)

        # Check budgets before recording
        await self._check_budgets(model, cost)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=cached_tokens,
            model=model,
        )

        record = CostRecord(
            model=model,
            tokens=usage,
            cost_usd=cost,
            task_id=task_id,
            agent_id=agent_id,
            tier=tier,
            metadata=metadata or {},
        )

        lock = await self._get_lock()
        async with lock:
            self._records.append(record)
            self._total_cost += cost
            self._total_tokens += usage.total_tokens
            self._total_calls += 1
            self._cost_by_model[model] += cost
            self._cost_by_tier[tier] += cost
            self._tokens_by_model[model] += usage.total_tokens

        logger.debug(
            "Recorded call: %s, tokens=%d, cost=$%.6f",
            model,
            usage.total_tokens,
            cost,
        )

        # Check alert thresholds
        await self._check_alerts(model)

        return record

    async def _check_budgets(self, model: str, attempted_cost: float) -> None:
        """Check if recording a call would exceed any budget.

        Args:
            model: The model being called.
            attempted_cost: The cost of the call.

        Raises:
            BudgetExceededError: If a hard-stop budget would be exceeded.
        """
        budgets = self._budgets.get(model, [])
        for budget in budgets:
            current = self._get_period_spend(model, budget.period)
            projected = current + attempted_cost

            if projected > budget.limit_usd and budget.hard_stop:
                raise BudgetExceededError(
                    model=model,
                    period=budget.period,
                    current_spend=current,
                    limit=budget.limit_usd,
                    attempted_cost=attempted_cost,
                )

    async def _check_alerts(self, model: str) -> None:
        """Check if any budget thresholds have been crossed.

        Args:
            model: The model to check.
        """
        budgets = self._budgets.get(model, [])
        for budget in budgets:
            current = self._get_period_spend(model, budget.period)
            pct = current / budget.limit_usd if budget.limit_usd > 0 else 0.0

            if pct >= budget.alert_threshold_pct:
                severity = (
                    AlertSeverity.CRITICAL
                    if pct >= 0.95
                    else AlertSeverity.WARNING
                    if pct >= 0.8
                    else AlertSeverity.INFO
                )

                alert = CostAlert(
                    model=model,
                    period=budget.period,
                    current_spend=current,
                    budget_limit=budget.limit_usd,
                    threshold_pct=pct,
                    severity=severity,
                    message=(
                        f"{model} {budget.period.value} budget: "
                        f"${current:.2f} / ${budget.limit_usd:.2f} "
                        f"({pct * 100:.1f}%)"
                    ),
                )
                self._alerts.append(alert)
                logger.warning("Cost alert: %s", alert.message)

                # Notify handlers
                for handler in self._alert_handlers:
                    try:
                        handler(alert)
                    except Exception as exc:
                        logger.error("Alert handler failed: %s", exc)

    def _get_period_spend(self, model: str, period: BudgetPeriod) -> float:
        """Get total spend for a model in a period.

        Args:
            model: The model name.
            period: The budget period.

        Returns:
            Total spend in USD.
        """
        now = time.time()
        period_start = self._period_start(now, period)

        total = 0.0
        for record in self._records:
            if record.model != model:
                continue
            if record.timestamp >= period_start:
                total += record.cost_usd
        return total

    def _period_start(self, now: float, period: BudgetPeriod) -> float:
        """Get the start timestamp for a budget period.

        Args:
            now: Current timestamp.
            period: The budget period.

        Returns:
            Start timestamp for the period.
        """
        if period == BudgetPeriod.HOURLY:
            return now - 3600
        if period == BudgetPeriod.DAILY:
            return now - 86400
        if period == BudgetPeriod.MONTHLY:
            return now - 30 * 86400
        return 0.0  # TOTAL

    # ------------------------------------------------------------------
    # Alert handlers
    # ------------------------------------------------------------------

    def add_alert_handler(self, handler: Callable[[CostAlert], Any]) -> None:
        """Register a handler for cost alerts.

        Args:
            handler: Callable that receives a CostAlert.
        """
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[CostAlert], Any]) -> None:
        """Remove an alert handler."""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_summary(self) -> CostSummary:
        """Get aggregated cost summary.

        Returns:
            CostSummary with totals and breakdowns.
        """
        avg_cost = self._total_cost / self._total_calls if self._total_calls > 0 else 0.0
        avg_tokens = self._total_tokens / self._total_calls if self._total_calls > 0 else 0.0

        by_period: dict[str, float] = defaultdict(float)
        for record in self._records:
            for period in BudgetPeriod:
                if record.timestamp >= self._period_start(time.time(), period):
                    by_period[period.value] += record.cost_usd

        return CostSummary(
            total_cost_usd=self._total_cost,
            total_tokens=self._total_tokens,
            total_calls=self._total_calls,
            by_model=dict(self._cost_by_model),
            by_tier={k: v for k, v in self._cost_by_tier.items()},
            by_period=dict(by_period),
            avg_cost_per_call=avg_cost,
            avg_tokens_per_call=avg_tokens,
        )

    def get_records(
        self,
        model: Optional[str] = None,
        tier: Optional[int] = None,
        limit: int = 100,
    ) -> list[CostRecord]:
        """Get cost records with optional filtering.

        Args:
            model: Filter by model name.
            tier: Filter by hierarchy tier.
            limit: Max records to return.

        Returns:
            List of cost records, most recent first.
        """
        records = self._records
        if model is not None:
            records = [r for r in records if r.model == model]
        if tier is not None:
            records = [r for r in records if r.tier == tier]
        return list(reversed(records[-limit:]))

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 50,
    ) -> list[CostAlert]:
        """Get cost alerts with optional filtering.

        Args:
            severity: Filter by severity level.
            limit: Max alerts to return.

        Returns:
            List of cost alerts, most recent first.
        """
        alerts = self._alerts
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        return list(reversed(alerts[-limit:]))

    def get_budget_status(self, model: str) -> list[dict[str, Any]]:
        """Get budget status for a model.

        Args:
            model: The model name.

        Returns:
            List of budget status dicts.
        """
        budgets = self._budgets.get(model, [])
        status: list[dict[str, Any]] = []
        for budget in budgets:
            current = self._get_period_spend(model, budget.period)
            pct = current / budget.limit_usd if budget.limit_usd > 0 else 0.0
            status.append({
                "model": model,
                "period": budget.period.value,
                "current_spend": current,
                "limit": budget.limit_usd,
                "remaining": max(0.0, budget.limit_usd - current),
                "pct_used": pct,
                "alert_threshold": budget.alert_threshold_pct,
                "hard_stop": budget.hard_stop,
            })
        return status

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Reset all cost records and counters (for testing)."""
        lock = await self._get_lock()
        async with lock:
            self._records.clear()
            self._alerts.clear()
            self._total_cost = 0.0
            self._total_tokens = 0
            self._total_calls = 0
            self._cost_by_model.clear()
            self._cost_by_tier.clear()
            self._tokens_by_model.clear()
        logger.info("Cost tracker reset")
