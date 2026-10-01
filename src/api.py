"""
Apex Harness API — FastAPI REST API.

Exposes the harness as a service with endpoints for:
  - Task routing (POST /route) — Laya-based routing via ApexGraphSwarm
  - Task execution (POST /execute) — Execute tasks on agents
  - Agent status (GET /status) — Real-time agent status
  - Memory operations (POST /memory/recall, /memory/remember, /memory/forget)
  - Cost reporting (GET /cost/report)
  - Governance checks (POST /governance/check)

Authentication: API key, OAuth2 + PKCE, mTLS
Authorization: RBAC + ABAC with tenant isolation
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    create_api_key,
    get_current_user,
    list_api_keys,
    handle_token_request,
    require_role,
    require_scope,
    revoke_api_key,
)
from integrations import IntegrationRegistry, IntegrationResult
from models import (
    AgentState,
    AgentStatus,
    CostBreakdown,
    CostReportResponse,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResponse,
    ExecuteResult,
    GovernanceCheckRequest,
    GovernanceCheckResponse,
    GovernanceFinding,
    GovernanceFramework,
    MemoryForgetRequest,
    MemoryForgetResponse,
    MemoryItem,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryRememberRequest,
    MemoryRememberResponse,
    Role,
    RouteRequest,
    RouteResponse,
    RouteDecision,
    StatusResponse,
    TaskStatus,
    TokenRequest,
    TokenResponse,
    UserContext,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("apex-harness")

# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------


class AppState:
    """Shared application state."""

    def __init__(self) -> None:
        self.registry: Optional[IntegrationRegistry] = None
        self.start_time: float = time.time()
        self._task_store: dict[str, dict] = {}  # In-memory task store
        self._memory_store: dict[str, list[dict]] = {}  # namespace -> memories
        self._cost_store: list[dict] = []  # Cost entries

    def get_registry(self) -> IntegrationRegistry:
        if self.registry is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Integrations not initialized",
            )
        return self.registry


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Apex Harness API...")
    app_state.registry = IntegrationRegistry.create_default()
    logger.info(f"Registered integrations: {app_state.registry.list_integrations()}")
    yield
    # Shutdown
    logger.info("Shutting down Apex Harness API...")
    if app_state.registry:
        await app_state.registry.close_all()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Apex Harness API",
    description="LLM-Agnostic Harness Engineering — API layer exposing the harness as a service.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standard error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=str(uuid.uuid4()),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            request_id=str(uuid.uuid4()),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Health & Root
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    integrations_healthy = {}
    if app_state.registry:
        integrations_healthy = {
            name: health.healthy
            for name, health in (await app_state.registry.health_check_all()).items()
        }

    return {
        "status": "healthy" if all(integrations_healthy.values()) else "degraded",
        "uptime_s": time.time() - app_state.start_time,
        "integrations": integrations_healthy,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/", tags=["System"])
async def root():
    """API root — basic info."""
    return {
        "name": "Apex Harness API",
        "version": "1.0.0",
        "description": "LLM-Agnostic Harness Engineering",
        "endpoints": [
            "POST /route",
            "POST /execute",
            "GET /status",
            "POST /memory/recall",
            "POST /memory/remember",
            "POST /memory/forget",
            "GET /cost/report",
            "POST /governance/check",
            "POST /auth/token",
            "POST /auth/api-keys",
        ],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
async def route_task(
    request: RouteRequest,
    user: UserContext = Depends(require_scope("route")),
):
    """
    Route a task to the most suitable agent using Laya (graph-based routing).

    The routing decision is made by ApexGraphSwarm's Laya router, which considers
    agent capabilities, current load, task type, and historical performance.
    """
    request_id = str(uuid.uuid4())
    registry = app_state.get_registry()
    swarm = registry.get("apex_graph_swarm")

    if not swarm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ApexGraphSwarm integration not available",
        )

    # Call the graph swarm router
    result = await swarm.route_task(
        task=request.task,
        context={
            **request.context,
            "task_type": request.task_type.value,
            "priority": request.priority.value,
            "preferred_agent": request.preferred_agent,
            "tenant_id": user.tenant_id,
        },
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Routing failed: {result.error}",
        )

    data = result.data
    decision = RouteDecision(
        agent_id=data.get("agent_id", "unknown"),
        agent_type=data.get("agent_type", "unknown"),
        confidence=data.get("confidence", 0.5),
        reasoning=data.get("reasoning", ""),
        estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
        estimated_duration_s=data.get("estimated_duration_s", 0.0),
    )

    alternatives = [
        RouteDecision(
            agent_id=alt.get("agent_id", "unknown"),
            agent_type=alt.get("agent_type", "unknown"),
            confidence=alt.get("confidence", 0.0),
            reasoning=alt.get("reasoning", ""),
            estimated_cost_usd=alt.get("estimated_cost_usd", 0.0),
            estimated_duration_s=alt.get("estimated_duration_s", 0.0),
        )
        for alt in data.get("alternatives", [])
    ]

    return RouteResponse(
        request_id=request_id,
        decision=decision,
        alternatives=alternatives,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@app.post("/execute", response_model=ExecuteResponse, tags=["Execution"])
async def execute_task(
    request: ExecuteRequest,
    user: UserContext = Depends(require_scope("execute")),
):
    """
    Execute a task on an agent.

    If `async_execution` is true, returns immediately with a job ID.
    Otherwise, waits for completion and returns the result.
    """
    request_id = str(uuid.uuid4())
    registry = app_state.get_registry()
    swarm = registry.get("apex_graph_swarm")

    if not swarm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ApexGraphSwarm integration not available",
        )

    # If no agent_id specified, route first
    agent_id = request.agent_id
    if not agent_id:
        route_result = await swarm.route_task(
            task=request.task,
            context={
                "task_type": request.task_type.value,
                "priority": request.priority.value,
                "tenant_id": user.tenant_id,
            },
        )
        if not route_result.success:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Routing failed: {route_result.error}",
            )
        agent_id = route_result.data.get("agent_id")

    job_id = str(uuid.uuid4())

    if request.async_execution:
        # Store task and return immediately
        app_state._task_store[job_id] = {
            "request_id": request_id,
            "agent_id": agent_id,
            "task": request.task,
            "status": TaskStatus.PENDING,
            "created_at": datetime.utcnow(),
            "tenant_id": user.tenant_id,
        }
        # Launch background execution
        asyncio.create_task(_execute_async(job_id, agent_id, request))
        return ExecuteResponse(
            request_id=request_id,
            job_id=job_id,
            status=TaskStatus.PENDING,
        )

    # Synchronous execution
    result = await swarm.dispatch_task(
        agent_id=agent_id,
        task=request.task,
        parameters={
            **request.parameters,
            "timeout_s": request.timeout_s,
            "tenant_id": user.tenant_id,
        },
    )

    if not result.success:
        return ExecuteResponse(
            request_id=request_id,
            job_id=job_id,
            status=TaskStatus.FAILED,
            error=result.error,
        )

    data = result.data
    return ExecuteResponse(
        request_id=request_id,
        job_id=job_id,
        status=TaskStatus.COMPLETED,
        result=ExecuteResult(
            output=data.get("output"),
            artifacts=data.get("artifacts", []),
            metrics=data.get("metrics", {}),
        ),
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


async def _execute_async(job_id: str, agent_id: str, request: ExecuteRequest) -> None:
    """Background task for async execution."""
    registry = app_state.get_registry()
    swarm = registry.get("apex_graph_swarm")
    if not swarm:
        return

    task_record = app_state._task_store.get(job_id)
    if not task_record:
        return

    task_record["status"] = TaskStatus.RUNNING
    try:
        result = await swarm.dispatch_task(
            agent_id=agent_id,
            task=request.task,
            parameters={
                **request.parameters,
                "timeout_s": request.timeout_s,
                "tenant_id": task_record["tenant_id"],
            },
        )
        if result.success:
            task_record["status"] = TaskStatus.COMPLETED
            task_record["result"] = result.data
        else:
            task_record["status"] = TaskStatus.FAILED
            task_record["error"] = result.error
    except Exception as exc:
        task_record["status"] = TaskStatus.FAILED
        task_record["error"] = str(exc)

    task_record["completed_at"] = datetime.utcnow()


@app.get("/execute/{job_id}", response_model=ExecuteResponse, tags=["Execution"])
async def get_execution_status(
    job_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get the status of an async execution job."""
    task_record = app_state._task_store.get(job_id)
    if not task_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    # Tenant isolation check
    if task_record.get("tenant_id") != user.tenant_id and user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    result = None
    if "result" in task_record:
        data = task_record["result"]
        result = ExecuteResult(
            output=data.get("output"),
            artifacts=data.get("artifacts", []),
            metrics=data.get("metrics", {}),
        )

    return ExecuteResponse(
        request_id=task_record.get("request_id", str(uuid.uuid4())),
        job_id=job_id,
        status=task_record.get("status", TaskStatus.PENDING),
        result=result,
        error=task_record.get("error"),
        started_at=task_record.get("created_at", datetime.utcnow()),
        completed_at=task_record.get("completed_at"),
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status(
    user: UserContext = Depends(get_current_user),
):
    """
    Get the status of all agents in the harness.

    Returns real-time status including state, current task, uptime,
    and task completion counts.
    """
    registry = app_state.get_registry()
    swarm = registry.get("apex_graph_swarm")

    if not swarm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ApexGraphSwarm integration not available",
        )

    result = await swarm.get_agents()
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get agent status: {result.error}",
        )

    agents = []
    for agent_data in result.data.get("agents", []):
        agents.append(
            AgentStatus(
                agent_id=agent_data.get("agent_id", "unknown"),
                agent_type=agent_data.get("agent_type", "unknown"),
                state=AgentState(agent_data.get("state", "idle")),
                current_task=agent_data.get("current_task"),
                uptime_s=agent_data.get("uptime_s", 0.0),
                tasks_completed=agent_data.get("tasks_completed", 0),
                tasks_failed=agent_data.get("tasks_failed", 0),
                last_heartbeat=agent_data.get("last_heartbeat"),
            )
        )

    healthy = sum(1 for a in agents if a.state in (AgentState.IDLE, AgentState.BUSY))

    return StatusResponse(
        agents=agents,
        total_agents=len(agents),
        healthy_agents=healthy,
    )


# ---------------------------------------------------------------------------
# Memory Operations
# ---------------------------------------------------------------------------


@app.post("/memory/recall", response_model=MemoryRecallResponse, tags=["Memory"])
async def memory_recall(
    request: MemoryRecallRequest,
    user: UserContext = Depends(require_scope("read")),
):
    """
    Recall memories based on a semantic query.

    Searches the memory store using similarity matching and returns
    the most relevant memories.
    """
    request_id = str(uuid.uuid4())
    namespace = user.tenant_id

    memories = app_state._memory_store.get(namespace, [])

    # Simple keyword-based recall (replace with vector similarity in production)
    query_lower = request.query.lower()
    scored = []
    for mem in memories:
        content_lower = mem["content"].lower()
        score = sum(1 for word in query_lower.split() if word in content_lower)
        if score > 0:
            scored.append((score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[: request.top_k]

    results = [
        MemoryItem(
            memory_id=mem["memory_id"],
            content=mem["content"],
            score=min(score / max(len(query_lower.split()), 1), 1.0),
            metadata=mem.get("metadata", {}),
            created_at=mem.get("created_at"),
        )
        for score, mem in top_results
    ]

    return MemoryRecallResponse(
        request_id=request_id,
        results=results,
        total_results=len(results),
    )


@app.post("/memory/remember", response_model=MemoryRememberResponse, tags=["Memory"])
async def memory_remember(
    request: MemoryRememberRequest,
    user: UserContext = Depends(require_scope("write")),
):
    """
    Store a new memory.

    Memories are stored in the specified namespace (defaults to tenant ID)
    and can be recalled later using semantic search.
    """
    request_id = str(uuid.uuid4())
    namespace = request.namespace or user.tenant_id

    if namespace not in app_state._memory_store:
        app_state._memory_store[namespace] = []

    memory_id = str(uuid.uuid4())
    memory_record = {
        "memory_id": memory_id,
        "content": request.content,
        "metadata": {
            **request.metadata,
            "tenant_id": user.tenant_id,
            "principal_id": user.principal_id,
        },
        "created_at": datetime.utcnow(),
        "ttl_seconds": request.ttl_seconds,
    }

    app_state._memory_store[namespace].append(memory_record)

    return MemoryRememberResponse(
        request_id=request_id,
        memory_id=memory_id,
        status="stored",
    )


@app.post("/memory/forget", response_model=MemoryForgetResponse, tags=["Memory"])
async def memory_forget(
    request: MemoryForgetRequest,
    user: UserContext = Depends(require_scope("write")),
):
    """
    Forget (delete) memories.

    Can forget by memory_id, by query match, or by namespace.
    """
    request_id = str(uuid.uuid4())
    forgotten_count = 0

    if request.memory_id:
        # Forget specific memory
        for namespace, memories in app_state._memory_store.items():
            before = len(memories)
            app_state._memory_store[namespace] = [
                m for m in memories if m["memory_id"] != request.memory_id
            ]
            forgotten_count += before - len(app_state._memory_store[namespace])

    elif request.query:
        # Forget memories matching query
        query_lower = request.query.lower()
        namespaces = [request.namespace] if request.namespace else list(app_state._memory_store.keys())
        for namespace in namespaces:
            if namespace not in app_state._memory_store:
                continue
            before = len(app_state._memory_store[namespace])
            app_state._memory_store[namespace] = [
                m for m in app_state._memory_store[namespace]
                if query_lower not in m["content"].lower()
            ]
            forgotten_count += before - len(app_state._memory_store[namespace])

    elif request.namespace:
        # Forget all memories in namespace
        if request.namespace in app_state._memory_store:
            forgotten_count = len(app_state._memory_store[request.namespace])
            app_state._memory_store[request.namespace] = []

    return MemoryForgetResponse(
        request_id=request_id,
        forgotten_count=forgotten_count,
        status="completed" if forgotten_count > 0 else "partial",
    )


# ---------------------------------------------------------------------------
# Cost Reporting
# ---------------------------------------------------------------------------


@app.get("/cost/report", response_model=CostReportResponse, tags=["Cost"])
async def cost_report(
    start: Optional[datetime] = Query(None, description="Start of reporting period"),
    end: Optional[datetime] = Query(None, description="End of reporting period"),
    user: UserContext = Depends(require_scope("read")),
):
    """
    Get a cost report for the specified period.

    Returns total cost, breakdown by category, and per-agent / per-tenant costs.
    """
    request_id = str(uuid.uuid4())

    # Filter by period
    entries = app_state._cost_store
    if start:
        entries = [e for e in entries if e["timestamp"] >= start]
    if end:
        entries = [e for e in entries if e["timestamp"] <= end]

    # Calculate totals
    total_cost = sum(e["amount_usd"] for e in entries)

    # Breakdown by category
    category_totals: dict[str, float] = {}
    agent_totals: dict[str, float] = {}
    tenant_totals: dict[str, float] = {}

    for entry in entries:
        cat = entry.get("category", "unknown")
        category_totals[cat] = category_totals.get(cat, 0.0) + entry["amount_usd"]

        agent = entry.get("agent_id", "unknown")
        agent_totals[agent] = agent_totals.get(agent, 0.0) + entry["amount_usd"]

        tenant = entry.get("tenant_id", "unknown")
        tenant_totals[tenant] = tenant_totals.get(tenant, 0.0) + entry["amount_usd"]

    breakdown = [
        CostBreakdown(
            category=cat,
            amount_usd=amount,
            percentage=(amount / total_cost * 100) if total_cost > 0 else 0.0,
        )
        for cat, amount in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    ]

    return CostReportResponse(
        request_id=request_id,
        total_cost_usd=total_cost,
        period_start=start,
        period_end=end,
        breakdown=breakdown,
        by_agent=agent_totals,
        by_tenant=tenant_totals,
    )


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


@app.post("/governance/check", response_model=GovernanceCheckResponse, tags=["Governance"])
async def governance_check(
    request: GovernanceCheckRequest,
    user: UserContext = Depends(require_scope("governance")),
):
    """
    Run a governance check against a target.

    Uses GRC_Claw to check compliance against ISO 42001, NIST AI RMF, or EU AI Act.
    """
    request_id = str(uuid.uuid4())
    registry = app_state.get_registry()
    grc = registry.get("grc_claw")

    if not grc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GRC_Claw integration not available",
        )

    result = await grc.check_compliance(
        framework=request.framework.value,
        target=request.target,
        context={
            **request.context,
            "strict": request.strict,
            "tenant_id": user.tenant_id,
        },
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Governance check failed: {result.error}",
        )

    data = result.data
    findings = [
        GovernanceFinding(
            rule_id=f.get("rule_id", "unknown"),
            severity=f.get("severity", "info"),
            message=f.get("message", ""),
            remediation=f.get("remediation"),
        )
        for f in data.get("findings", [])
    ]

    # Determine pass/fail
    has_violations = any(f.severity in ("violation", "critical") for f in findings)
    has_warnings = any(f.severity == "warning" for f in findings)
    passed = not has_violations and not (request.strict and has_warnings)

    return GovernanceCheckResponse(
        request_id=request_id,
        framework=request.framework,
        target=request.target,
        passed=passed,
        score=data.get("score", 100.0 if passed else 0.0),
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
async def create_token(request: TokenRequest):
    """
    OAuth2 token endpoint.

    Supports client_credentials, authorization_code, and refresh_token grants.
    """
    return handle_token_request(request)


@app.post("/auth/api-keys", response_model=ApiKeyCreateResponse, tags=["Auth"])
async def create_api_key_endpoint(
    request: ApiKeyCreateRequest,
    user: UserContext = Depends(require_role(Role.ADMIN)),
):
    """
    Create a new API key.

    Requires admin role. The raw API key is returned only once.
    """
    return create_api_key(request)


@app.get("/auth/api-keys", tags=["Auth"])
async def list_api_keys_endpoint(
    user: UserContext = Depends(require_role(Role.ADMIN)),
):
    """List all API keys (admin only)."""
    return list_api_keys()


@app.delete("/auth/api-keys/{key_id}", tags=["Auth"])
async def revoke_api_key_endpoint(
    key_id: str,
    user: UserContext = Depends(require_role(Role.ADMIN)),
):
    """Revoke an API key (admin only)."""
    if not revoke_api_key(key_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        )
    return {"status": "revoked", "key_id": key_id}


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
