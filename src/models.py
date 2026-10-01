"""
Pydantic models for the Apex Harness API.

All request/response schemas for the REST API, including task routing,
execution, memory operations, cost reporting, and governance checks.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Supported task types for routing."""

    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"
    ORCHESTRATION = "orchestration"
    GOVERNANCE = "governance"
    MEMORY = "memory"


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Possible task execution statuses."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryOperation(str, Enum):
    """Memory operation types."""

    RECALL = "recall"
    REMEMBER = "remember"
    FORGET = "forget"


class GovernanceFramework(str, Enum):
    """Supported governance frameworks."""

    ISO_42001 = "iso_42001"
    NIST_AI_RMF = "nist_ai_rmf"
    EU_AI_ACT = "eu_ai_act"


class AgentState(str, Enum):
    """Possible agent states."""

    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class AuthScheme(str, Enum):
    """Supported authentication schemes."""

    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    MTLS = "mtls"


class Role(str, Enum):
    """RBAC roles."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SERVICE = "service"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    """Request body for POST /route."""

    task: str = Field(..., min_length=1, max_length=4096, description="Natural-language task description")
    task_type: TaskType = Field(default=TaskType.ANALYSIS, description="Category of the task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Execution priority")
    context: dict[str, Any] = Field(default_factory=dict, description="Arbitrary context key-value pairs")
    preferred_agent: Optional[str] = Field(default=None, description="Hint for a specific agent to use")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Caller-supplied metadata")


class RouteDecision(BaseModel):
    """Routing decision returned by the Laya router."""

    agent_id: str = Field(..., description="Selected agent identifier")
    agent_type: str = Field(..., description="Agent type / capability label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence score")
    reasoning: str = Field(..., description="Human-readable routing rationale")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD")
    estimated_duration_s: float = Field(default=0.0, ge=0.0, description="Estimated duration in seconds")


class RouteResponse(BaseModel):
    """Response body for POST /route."""

    request_id: str = Field(..., description="Unique request identifier (UUID)")
    decision: RouteDecision = Field(..., description="The routing decision")
    alternatives: list[RouteDecision] = Field(default_factory=list, description="Alternative agents considered")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    """Request body for POST /execute."""

    task: str = Field(..., min_length=1, max_length=4096, description="Task to execute")
    task_type: TaskType = Field(default=TaskType.ANALYSIS)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    agent_id: Optional[str] = Field(default=None, description="Override the routed agent")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    timeout_s: float = Field(default=300.0, gt=0, le=3600, description="Execution timeout in seconds")
    async_execution: bool = Field(default=False, description="If true, return immediately with a job ID")


class ExecuteResult(BaseModel):
    """Result payload from task execution."""

    output: Any = Field(..., description="Arbitrary execution output")
    artifacts: list[str] = Field(default_factory=list, description="Paths or URIs of produced artifacts")
    metrics: dict[str, float] = Field(default_factory=dict, description="Execution metrics (tokens, latency, etc.)")


class ExecuteResponse(BaseModel):
    """Response body for POST /execute."""

    request_id: str = Field(..., description="Unique request identifier (UUID)")
    job_id: Optional[str] = Field(default=None, description="Job ID for async execution")
    status: TaskStatus = Field(..., description="Current execution status")
    result: Optional[ExecuteResult] = Field(default=None, description="Execution result (null if pending)")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class AgentStatus(BaseModel):
    """Status of a single agent."""

    agent_id: str
    agent_type: str
    state: AgentState
    current_task: Optional[str] = None
    uptime_s: float = Field(default=0.0, ge=0.0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    last_heartbeat: Optional[datetime] = None


class StatusResponse(BaseModel):
    """Response body for GET /status."""

    agents: list[AgentStatus] = Field(default_factory=list)
    total_agents: int = Field(default=0)
    healthy_agents: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryRecallRequest(BaseModel):
    """Request body for POST /memory/recall."""

    query: str = Field(..., min_length=1, max_length=2048, description="Recall query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    include_embeddings: bool = Field(default=False, description="Include embedding vectors in response")


class MemoryItem(BaseModel):
    """A single memory entry."""

    memory_id: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class MemoryRecallResponse(BaseModel):
    """Response body for POST /memory/recall."""

    request_id: str
    results: list[MemoryItem] = Field(default_factory=list)
    total_results: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryRememberRequest(BaseModel):
    """Request body for POST /memory/remember."""

    content: str = Field(..., min_length=1, max_length=65536, description="Content to store")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    namespace: str = Field(default="default", description="Memory namespace / tenant scope")
    ttl_seconds: Optional[int] = Field(default=None, ge=1, description="Time-to-live in seconds")


class MemoryRememberResponse(BaseModel):
    """Response body for POST /memory/remember."""

    request_id: str
    memory_id: str
    status: Literal["stored", "deduplicated", "failed"] = "stored"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryForgetRequest(BaseModel):
    """Request body for POST /memory/forget."""

    memory_id: Optional[str] = Field(default=None, description="Specific memory ID to forget")
    query: Optional[str] = Field(default=None, description="Forget memories matching this query")
    namespace: Optional[str] = Field(default=None, description="Forget all memories in this namespace")
    filters: dict[str, Any] = Field(default_factory=dict, description="Additional filters")


class MemoryForgetResponse(BaseModel):
    """Response body for POST /memory/forget."""

    request_id: str
    forgotten_count: int = Field(default=0, ge=0)
    status: Literal["completed", "partial", "failed"] = "completed"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


class CostBreakdown(BaseModel):
    """Cost breakdown by category."""

    category: str
    amount_usd: float = Field(default=0.0, ge=0.0)
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)


class CostReportResponse(BaseModel):
    """Response body for GET /cost/report."""

    request_id: str
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    breakdown: list[CostBreakdown] = Field(default_factory=list)
    by_agent: dict[str, float] = Field(default_factory=dict)
    by_tenant: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


class GovernanceCheckRequest(BaseModel):
    """Request body for POST /governance/check."""

    framework: GovernanceFramework = Field(default=GovernanceFramework.ISO_42001)
    target: str = Field(..., min_length=1, description="Agent, model, or process to check")
    context: dict[str, Any] = Field(default_factory=dict)
    strict: bool = Field(default=False, description="Fail on warnings, not just violations")


class GovernanceFinding(BaseModel):
    """A single governance finding."""

    rule_id: str
    severity: Literal["info", "warning", "violation", "critical"]
    message: str
    remediation: Optional[str] = None


class GovernanceCheckResponse(BaseModel):
    """Response body for POST /governance/check."""

    request_id: str
    framework: GovernanceFramework
    target: str
    passed: bool
    score: float = Field(..., ge=0.0, le=100.0)
    findings: list[GovernanceFinding] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Auth / RBAC
# ---------------------------------------------------------------------------


class TokenRequest(BaseModel):
    """OAuth2 token request."""

    grant_type: Literal["client_credentials", "authorization_code", "refresh_token"]
    client_id: str
    client_secret: Optional[str] = None
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class TokenResponse(BaseModel):
    """OAuth2 token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class ApiKeyCreateRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=128)
    role: Role = Field(default=Role.VIEWER)
    tenant_id: str = Field(default="default")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)
    scopes: list[str] = Field(default_factory=list)


class ApiKeyCreateResponse(BaseModel):
    """Response with the newly created API key (shown only once)."""

    key_id: str
    api_key: str = Field(..., description="The raw API key — store securely, shown only once")
    name: str
    role: Role
    tenant_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class ApiKeyInfo(BaseModel):
    """Public API key metadata (no secret)."""

    key_id: str
    name: str
    role: Role
    tenant_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class UserContext(BaseModel):
    """Authenticated user / service principal context."""

    principal_id: str
    role: Role
    tenant_id: str
    auth_scheme: AuthScheme
    scopes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
