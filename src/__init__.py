"""Apex Harness — LLM-Agnostic Harness Engineering.

Unified orchestration layer for multi-agent swarms with hierarchical
delegation, cost-aware model routing, and evidence-backed completion.
"""

from .orchestrator import HierarchicalOrchestrator, Tier, Task, TaskResult
from .router import ModelRouter, RoutingDecision, LayaClient
from .scheduler import Scheduler, AgentSlot, Wave, WaveResult
from .cost_tracker import CostTracker, BudgetExceededError, TokenUsage

__all__ = [
    "HierarchicalOrchestrator",
    "Tier",
    "Task",
    "TaskResult",
    "ModelRouter",
    "RoutingDecision",
    "LayaClient",
    "Scheduler",
    "AgentSlot",
    "Wave",
    "WaveResult",
    "CostTracker",
    "BudgetExceededError",
    "TokenUsage",
]

__version__ = "0.1.0"
